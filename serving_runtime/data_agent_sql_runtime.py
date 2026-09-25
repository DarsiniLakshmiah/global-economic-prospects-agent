# ============================================================
# Serving-compatible deterministic Data Agent
# ============================================================
#
# No Spark session is required.
#
# Structured historical facts are read from:
#
#   worldbank_ai.gold.macroeconomic_indicators
#
# through:
#
#   Databricks Statement Execution API
#       -> SQL Warehouse
#       -> Unity Catalog Gold table
#
# The LLM is NOT allowed to execute arbitrary SQL.
# Only deterministic, parameterized functions are exposed.
# ============================================================


from typing import Any, Dict, List

from databricks.sdk import WorkspaceClient

from databricks.sdk.service.sql import (
    StatementParameterListItem,
    Disposition,
    Format,
    ExecuteStatementRequestOnWaitTimeout,
)

import json
import math
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

MACRO_TABLE = "worldbank_ai.gold.macroeconomic_indicators"

MIN_OBSERVATION_YEAR = 2010

# Historical World Bank indicator observations currently stop
# at 2025 in our governed dataset.
#
# 2026 forecasts must go through the GEP Research Agent.
MAX_OBSERVATION_YEAR = 2025

MAX_YEAR_SPAN = 30

MAX_ENTITIES = 10


# ============================================================
# SUPPORTED INDICATORS
# ============================================================
#
# Only these governed indicators can be requested by the
# structured Data Agent.
#
# This prevents arbitrary indicator/tool execution.
# ============================================================

INDICATOR_REGISTRY = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (annual %)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
    "NE.GDI.TOTL.ZS": "Gross capital formation (% of GDP)",
    "GC.XPN.TOTL.GD.ZS": "Expense (% of GDP)",
    "GC.REV.XGRT.GD.ZS": "Revenue, excluding grants (% of GDP)",
    "GC.DOD.TOTL.GD.ZS": "Central government debt, total (% of GDP)",
    "NY.GDS.TOTL.ZS": "Gross domestic savings (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
}


# ============================================================
# JSON SAFETY
# ============================================================

def _json_safe(value: Any) -> Any:
    """
    Convert returned values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]

    return str(value)


# ============================================================
# SQL STORE
# ============================================================

class SQLStructuredStore:
    """
    Serving-compatible access layer for structured data.

    Development notebooks can use Spark directly.

    Custom Model Serving does not depend on the notebook Spark
    session, so production structured queries are executed
    through a Databricks SQL Warehouse.
    """

    def __init__(
        self,
        warehouse_id: str | None = None,
    ):

        self.warehouse_id = (
            warehouse_id
            or os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
        )

        if not self.warehouse_id:
            raise RuntimeError(
                "DATABRICKS_SQL_WAREHOUSE_ID is required "
                "for serving-time structured queries."
            )

        # Uses Databricks-native authentication.
        # No PAT is hardcoded here.
        self.w = WorkspaceClient()


    def query(
        self,
        statement: str,
        parameters: List[StatementParameterListItem] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute parameterized SQL through the configured
        Databricks SQL Warehouse.
        """

        # ----------------------------------------------------
        # Execute statement
        # ----------------------------------------------------
        #
        # These are SDK enums, not raw strings.
        #
        # Using strings such as:
        #
        #   "CANCEL"
        #   "INLINE"
        #   "JSON_ARRAY"
        #
        # can cause:
        #
        #   AttributeError:
        #   'str' object has no attribute 'value'
        #
        # ----------------------------------------------------

        response = (
            self.w.statement_execution.execute_statement(
                warehouse_id=self.warehouse_id,
                statement=statement,
                parameters=parameters or [],

                wait_timeout="30s",

                on_wait_timeout=(
                    ExecuteStatementRequestOnWaitTimeout.CANCEL
                ),

                disposition=Disposition.INLINE,

                format=Format.JSON_ARRAY,
            )
        )


        # ----------------------------------------------------
        # Validate execution status
        # ----------------------------------------------------

        if response.status is None:
            raise RuntimeError(
                "SQL statement returned no execution status."
            )

        state = str(response.status.state)

        if "SUCCEEDED" not in state:

            message = None

            if response.status.error is not None:
                message = response.status.error.message

            raise RuntimeError(
                f"SQL statement failed: {state}. "
                f"{message or ''}".strip()
            )


        # ----------------------------------------------------
        # Validate result metadata
        # ----------------------------------------------------

        if (
            response.manifest is None
            or response.manifest.schema is None
            or response.manifest.schema.columns is None
        ):
            return []


        columns = [
            column.name
            for column
            in response.manifest.schema.columns
        ]


        # ----------------------------------------------------
        # Extract returned rows
        # ----------------------------------------------------

        if response.result is None:
            return []

        rows = response.result.data_array or []


        # ----------------------------------------------------
        # Convert rows to dictionaries
        # ----------------------------------------------------

        results = []

        for row in rows:

            record = dict(
                zip(
                    columns,
                    row,
                )
            )

            results.append(
                _json_safe(record)
            )

        return results


# ============================================================
# LAZY SQL STORE
# ============================================================
#
# Do NOT create the WorkspaceClient / warehouse connection
# during module import.
#
# The store is initialized only when the Data Agent actually
# needs structured data.
# ============================================================

_store = None


def _get_store() -> SQLStructuredStore:
    """
    Lazily initialize one SQLStructuredStore per Python process.
    """

    global _store

    if _store is None:
        _store = SQLStructuredStore()

    return _store


# ============================================================
# INDICATOR VALIDATION
# ============================================================

def validate_indicator(
    indicator_code: str,
) -> str:
    """
    Allow only indicators in our governed registry.
    """

    if indicator_code not in INDICATOR_REGISTRY:
        raise ValueError(
            f"Unsupported indicator: {indicator_code}."
        )

    return indicator_code


# ============================================================
# YEAR VALIDATION
# ============================================================

def validate_year_range(
    start_year: int,
    end_year: int,
):
    """
    Validate the historical observation window.

    Forecast years are intentionally excluded from this tool.
    """

    start_year = int(start_year)
    end_year = int(end_year)

    if start_year < MIN_OBSERVATION_YEAR:
        raise ValueError(
            f"start_year must be >= "
            f"{MIN_OBSERVATION_YEAR}."
        )

    if end_year > MAX_OBSERVATION_YEAR:
        raise ValueError(
            f"Historical observations stop at "
            f"{MAX_OBSERVATION_YEAR}; "
            "forecasts must use the GEP research path."
        )

    if start_year > end_year:
        raise ValueError(
            "start_year cannot be greater than end_year."
        )

    if (
        end_year
        - start_year
        + 1
        > MAX_YEAR_SPAN
    ):
        raise ValueError(
            f"Requested range exceeds "
            f"{MAX_YEAR_SPAN} years."
        )

    return start_year, end_year


# ============================================================
# ENTITY SEARCH
# ============================================================

def search_entities(
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search governed entities by:

    - entity name
    - ISO3/entity ID
    - ISO2 code

    Search is parameterized.
    """

    query = str(query).strip()

    if not query:
        raise ValueError(
            "Entity search query cannot be empty."
        )

    limit = max(
        1,
        min(
            int(limit),
            20,
        ),
    )


    sql = f"""
    WITH entities AS (
        SELECT DISTINCT
            entity_id,
            iso2_code,
            entity_name,
            entity_type,
            region_id,
            region_name,
            income_level_name
        FROM {MACRO_TABLE}
    )

    SELECT *
    FROM entities

    WHERE
        lower(entity_name)
            LIKE concat('%', lower(:q), '%')

        OR lower(entity_id) = lower(:q)

        OR lower(iso2_code) = lower(:q)

    ORDER BY

        CASE

            WHEN lower(entity_name) = lower(:q)
                THEN 0

            WHEN lower(entity_id) = lower(:q)
                THEN 1

            WHEN lower(iso2_code) = lower(:q)
                THEN 2

            ELSE 3

        END,

        length(entity_name),

        entity_name

    LIMIT {limit}
    """


    return _get_store().query(
        sql,
        [
            StatementParameterListItem(
                name="q",
                value=query,
                type="STRING",
            )
        ],
    )


# ============================================================
# ENTITY RESOLUTION
# ============================================================

def resolve_entity(
    entity_query: str,
) -> Dict[str, Any]:
    """
    Resolve a country/economy/aggregate to a canonical entity.

    Exact resolution is required before structured execution.
    """

    entity_query = str(
        entity_query
    ).strip()

    if not entity_query:
        raise ValueError(
            "entity_query cannot be empty."
        )


    sql = f"""
    SELECT DISTINCT

        entity_id,
        iso2_code,
        entity_name,
        entity_type,
        region_id,
        region_name,
        income_level_name

    FROM {MACRO_TABLE}

    WHERE

        lower(entity_name) = lower(:q)

        OR lower(entity_id) = lower(:q)

        OR lower(iso2_code) = lower(:q)
    """


    exact = _get_store().query(
        sql,
        [
            StatementParameterListItem(
                name="q",
                value=entity_query,
                type="STRING",
            )
        ],
    )


    # Exactly one canonical match.
    if len(exact) == 1:
        return exact[0]


    # Multiple exact matches should not be guessed.
    if len(exact) > 1:
        raise ValueError(
            f"Entity is ambiguous for "
            f"'{entity_query}': "
            f"{json.dumps(exact, ensure_ascii=False)}"
        )


    # Return candidate information in the error,
    # but do not silently choose one.
    candidates = search_entities(
        entity_query,
        10,
    )

    raise ValueError(
        f"Could not resolve entity "
        f"'{entity_query}' exactly. "
        f"Candidates: "
        f"{json.dumps(candidates, ensure_ascii=False)}"
    )


# ============================================================
# INDICATOR SERIES
# ============================================================

def get_indicator_series(
    entity_query: str,
    indicator_code: str,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    """
    Return one historical indicator series.

    Missing numeric values remain None/null.
    """

    indicator_code = validate_indicator(
        indicator_code
    )

    (
        start_year,
        end_year,
    ) = validate_year_range(
        start_year,
        end_year,
    )

    entity = resolve_entity(
        entity_query
    )


    sql = f"""
    SELECT

        year,
        value,
        has_value,
        unit_type,
        unit_label,
        value_type,
        source_system

    FROM {MACRO_TABLE}

    WHERE entity_id = :entity_id

      AND indicator_id = :indicator_id

      AND year BETWEEN
          :start_year
          AND :end_year

    ORDER BY year
    """


    rows = _get_store().query(
        sql,
        [
            StatementParameterListItem(
                name="entity_id",
                value=str(
                    entity["entity_id"]
                ),
                type="STRING",
            ),

            StatementParameterListItem(
                name="indicator_id",
                value=indicator_code,
                type="STRING",
            ),

            StatementParameterListItem(
                name="start_year",
                value=str(start_year),
                type="INT",
            ),

            StatementParameterListItem(
                name="end_year",
                value=str(end_year),
                type="INT",
            ),
        ],
    )


    # --------------------------------------------------------
    # Normalize Statement Execution JSON values
    # --------------------------------------------------------
    #
    # Statement Execution JSON values can arrive as strings.
    # Normalize known scalar types while preserving nulls.
    # --------------------------------------------------------

    observations = []

    for row in rows:

        value = row.get(
            "value"
        )

        observations.append(
            {
                **row,

                "year": (
                    int(row["year"])
                    if row.get("year") is not None
                    else None
                ),

                "value": (
                    float(value)
                    if value not in (None, "")
                    else None
                ),

                "has_value": (
                    str(
                        row.get("has_value")
                    ).lower()
                    == "true"
                ),
            }
        )


    observed_count = sum(
        1
        for row in observations
        if row["value"] is not None
    )


    return {

        "entity": entity,

        "indicator_code": indicator_code,

        "indicator_name": (
            INDICATOR_REGISTRY[
                indicator_code
            ]
        ),

        "start_year": start_year,

        "end_year": end_year,

        "observations": observations,

        "observation_count": len(
            observations
        ),

        "non_null_observation_count": (
            observed_count
        ),

        "missing_year_count": (
            end_year
            - start_year
            + 1
            - observed_count
        ),

        "historical_only": True,
    }


# ============================================================
# STRUCTURED PLAN VALIDATION
# ============================================================

def validate_structured_plan(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the Supervisor -> Data Agent contract.
    """

    if not isinstance(
        plan,
        dict,
    ):
        raise TypeError(
            "Supervisor plan must be a dictionary."
        )


    route = plan.get(
        "route"
    )

    if route not in {
        "structured",
        "hybrid",
    }:
        raise ValueError(
            f"Data Agent cannot execute "
            f"route={route!r}."
        )


    if (
        plan.get(
            "needs_structured_tool"
        )
        is not True
    ):
        raise ValueError(
            "Supervisor did not authorize "
            "structured execution."
        )


    entities = (
        plan.get(
            "countries_or_entities"
        )
        or []
    )

    indicators = (
        plan.get(
            "indicators"
        )
        or []
    )

    start_year = plan.get(
        "observation_start_year"
    )

    end_year = plan.get(
        "observation_end_year"
    )


    if not entities:
        raise ValueError(
            "Structured execution requires "
            "at least one entity."
        )


    if len(entities) > MAX_ENTITIES:
        raise ValueError(
            f"Maximum entities is "
            f"{MAX_ENTITIES}."
        )


    if not indicators:
        raise ValueError(
            "Structured execution requires "
            "at least one indicator."
        )


    for code in indicators:
        validate_indicator(
            code
        )


    if start_year is None:
        raise ValueError(
            "observation_start_year is required."
        )


    if end_year is None:
        end_year = (
            MAX_OBSERVATION_YEAR
        )


    (
        start_year,
        end_year,
    ) = validate_year_range(
        start_year,
        end_year,
    )


    return {

        "route": route,

        "resolved_query": (
            plan.get(
                "resolved_query",
                plan.get(
                    "original_query",
                    "",
                ),
            )
        ),

        "entities": [
            str(x).strip()
            for x in entities
            if str(x).strip()
        ],

        "indicators": list(
            dict.fromkeys(
                indicators
            )
        ),

        "start_year": start_year,

        "end_year": end_year,
    }


# ============================================================
# DATA AGENT
# ============================================================

def run_data_agent(
    supervisor_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the structured portion of an approved Supervisor plan.
    """

    start = time.perf_counter()


    # Validate Supervisor -> Data Agent contract.
    plan = validate_structured_plan(
        supervisor_plan
    )


    # --------------------------------------------------------
    # Resolve requested entities
    # --------------------------------------------------------

    resolved_entities = []

    for requested in plan["entities"]:

        resolved_entities.append(
            {
                "requested_name": requested,

                "resolved_entity": (
                    resolve_entity(
                        requested
                    )
                ),
            }
        )


    # --------------------------------------------------------
    # Retrieve requested indicator series
    # --------------------------------------------------------

    series = []

    for item in resolved_entities:

        canonical = (
            item[
                "resolved_entity"
            ][
                "entity_name"
            ]
        )

        for code in plan["indicators"]:

            series.append(
                get_indicator_series(
                    canonical,
                    code,
                    plan["start_year"],
                    plan["end_year"],
                )
            )


    # --------------------------------------------------------
    # Return governed Data Agent contract
    # --------------------------------------------------------

    return {

        "agent": "data_agent",

        "status": "success",

        "route": plan["route"],

        "query": (
            plan["resolved_query"]
        ),

        "historical_observation_window": {

            "start_year": (
                plan["start_year"]
            ),

            "end_year": (
                plan["end_year"]
            ),
        },

        "resolved_entities": (
            resolved_entities
        ),

        "indicator_count": len(
            plan["indicators"]
        ),

        "series_count": len(
            series
        ),

        "series": series,

        "execution_latency_ms": round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            2,
        ),

        "data_contract": {

            "source": MACRO_TABLE,

            "historical_only": True,

            "max_historical_year": (
                MAX_OBSERVATION_YEAR
            ),

            "missing_values_preserved_as_null": True,

            "arbitrary_sql_exposed": False,

            "serving_access": (
                "parameterized_statement_execution"
            ),
        },
    }


# ============================================================
# SYNTHESIS PAYLOAD
# ============================================================

def build_structured_synthesis_payload(
    data_agent_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the compact structured evidence payload consumed by
    the Synthesis Agent.
    """

    if (
        data_agent_result.get(
            "status"
        )
        != "success"
    ):
        raise ValueError(
            "Cannot synthesize a failed "
            "Data Agent result."
        )


    compact = []


    for series in data_agent_result[
        "series"
    ]:

        compact.append(
            {

                "entity_id": (
                    series[
                        "entity"
                    ][
                        "entity_id"
                    ]
                ),

                "entity_name": (
                    series[
                        "entity"
                    ][
                        "entity_name"
                    ]
                ),

                "entity_type": (
                    series[
                        "entity"
                    ][
                        "entity_type"
                    ]
                ),

                "indicator_code": (
                    series[
                        "indicator_code"
                    ]
                ),

                "indicator_name": (
                    series[
                        "indicator_name"
                    ]
                ),

                "start_year": (
                    series[
                        "start_year"
                    ]
                ),

                "end_year": (
                    series[
                        "end_year"
                    ]
                ),

                "observations": [

                    {
                        "year": (
                            obs["year"]
                        ),

                        "value": (
                            obs["value"]
                        ),

                        "unit_label": (
                            obs["unit_label"]
                        ),
                    }

                    for obs
                    in series[
                        "observations"
                    ]
                ],

                "missing_year_count": (
                    series[
                        "missing_year_count"
                    ]
                ),
            }
        )


    return {

        "evidence_type": (
            "structured_historical"
        ),

        "query": (
            data_agent_result[
                "query"
            ]
        ),

        "source_table": (
            MACRO_TABLE
        ),

        "historical_only": True,

        "series": compact,
    }