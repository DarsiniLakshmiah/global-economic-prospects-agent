# Databricks notebook source
# MAGIC %md
# MAGIC # runtime / data_agent_runtime
# MAGIC
# MAGIC Definitions only. No package installation, Python restart, smoke test,
# MAGIC benchmark execution, or development validation cells.
# MAGIC
# MAGIC This notebook is safe to import with `%run` from evaluation/serving notebooks.

# COMMAND ----------

# MAGIC ## 01. Imports and configuration

# COMMAND ----------

from typing import Any, Dict, List, Optional
from pyspark.sql import functions as F
import json
import math
import time

CATALOG = "worldbank_ai"
GOLD_SCHEMA = "gold"

MACRO_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.macroeconomic_indicators"

MIN_OBSERVATION_YEAR = 2010
MAX_OBSERVATION_YEAR = 2025
MAX_YEAR_SPAN = 30
MAX_ENTITIES = 20

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

print("Data Agent configuration loaded.")
print("Structured source:", MACRO_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Preflight: validate the locked Gold contract
# MAGIC
# MAGIC The physical Gold schema uses:
# MAGIC - `indicator_id` for the World Bank indicator code
# MAGIC - `indicator_display_name` for the readable name
# MAGIC
# MAGIC We intentionally do **not** expect physical columns named `indicator_code` or `indicator_name`.

# COMMAND ----------

if not spark.catalog.tableExists(MACRO_TABLE):
    raise ValueError(
        f"Required Gold table does not exist: {MACRO_TABLE}"
    )

macro_df = spark.table(MACRO_TABLE)

REQUIRED_COLUMNS = {
    "entity_id",
    "iso2_code",
    "entity_name",
    "entity_type",
    "indicator_id",
    "indicator_display_name",
    "indicator_category",
    "year",
    "value",
    "unit_type",
    "unit_label",
    "value_type",
    "has_value",
}

missing_columns = sorted(
    REQUIRED_COLUMNS - set(macro_df.columns)
)

if missing_columns:
    raise ValueError(
        "Gold macro table does not match the Data Agent contract. "
        f"Missing columns: {missing_columns}"
    )

print("Gold schema validation passed.")
print("Structured source:", MACRO_TABLE)

# COMMAND ----------

# MAGIC ## 03. JSON-safe conversion helpers

# COMMAND ----------

def json_safe(value: Any) -> Any:
    """Convert Spark/Python values into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    # Decimal and similar numeric values.
    try:
        if hasattr(value, "as_integer_ratio"):
            return float(value)
    except Exception:
        pass

    return str(value)


def rows_to_dicts(rows) -> List[Dict[str, Any]]:
    """Convert Spark Row objects into JSON-safe dictionaries."""
    return [
        json_safe(row.asDict(recursive=True))
        for row in rows
    ]


print("JSON-safe helpers loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Validation helpers

# COMMAND ----------

def validate_indicator(
    indicator_code: str,
) -> str:
    """Allow only indicators exposed by the governed registry."""
    if indicator_code not in INDICATOR_REGISTRY:
        raise ValueError(
            f"Unsupported indicator: {indicator_code}. "
            "The Data Agent can only use the governed indicator registry."
        )

    return indicator_code


def validate_year_range(
    start_year: int,
    end_year: int,
) -> tuple:
    """
    Validate historical observation years.

    2026+ is rejected rather than silently interpreted as historical data.
    """
    start_year = int(start_year)
    end_year = int(end_year)

    if start_year < MIN_OBSERVATION_YEAR:
        raise ValueError(
            f"start_year must be >= {MIN_OBSERVATION_YEAR}."
        )

    if end_year > MAX_OBSERVATION_YEAR:
        raise ValueError(
            f"Historical observations stop at {MAX_OBSERVATION_YEAR}. "
            f"Requested end_year={end_year}. "
            "Forecasts/projections must use the GEP research path."
        )

    if start_year > end_year:
        raise ValueError(
            "start_year cannot be greater than end_year."
        )

    if end_year - start_year + 1 > MAX_YEAR_SPAN:
        raise ValueError(
            f"Requested range exceeds the maximum {MAX_YEAR_SPAN}-year span."
        )

    return start_year, end_year


print("Validation helpers loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Safe entity search and resolution
# MAGIC
# MAGIC Resolution policy:
# MAGIC 1. exact case-insensitive `entity_name`
# MAGIC 2. exact `entity_id`
# MAGIC 3. exact `iso2_code`
# MAGIC 4. otherwise return candidates and **do not silently choose**
# MAGIC
# MAGIC This is important because the Gold table contains both countries/economies and World Bank aggregates.

# COMMAND ----------

def search_entities(
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return governed entity candidates without choosing one."""
    query = str(query).strip()

    if not query:
        raise ValueError("Entity search query cannot be empty.")

    limit = max(1, min(int(limit), 20))

    entity_df = (
        macro_df
        .select(
            "entity_id",
            "iso2_code",
            "entity_name",
            "entity_type",
            "region_id",
            "region_name",
            "income_level_name",
        )
        .dropDuplicates(
            ["entity_id", "entity_name", "entity_type"]
        )
    )

    q = query.lower()

    result = (
        entity_df
        .filter(
            F.lower(F.col("entity_name")).contains(q)
            | (F.lower(F.col("entity_id")) == q)
            | (F.lower(F.col("iso2_code")) == q)
        )
        .orderBy(
            F.when(
                F.lower(F.col("entity_name")) == q,
                F.lit(0),
            )
            .when(
                F.lower(F.col("entity_id")) == q,
                F.lit(1),
            )
            .when(
                F.lower(F.col("iso2_code")) == q,
                F.lit(2),
            )
            .otherwise(F.lit(3)),
            F.length("entity_name"),
            "entity_name",
        )
        .limit(limit)
        .collect()
    )

    return rows_to_dicts(result)


def resolve_entity(
    entity_query: str,
) -> Dict[str, Any]:
    """
    Resolve one entity only when the match is unambiguous.

    No fuzzy candidate is silently selected.
    """
    entity_query = str(entity_query).strip()

    if not entity_query:
        raise ValueError("entity_query cannot be empty.")

    entity_df = (
        macro_df
        .select(
            "entity_id",
            "iso2_code",
            "entity_name",
            "entity_type",
            "region_id",
            "region_name",
            "income_level_name",
        )
        .dropDuplicates(
            ["entity_id", "entity_name", "entity_type"]
        )
    )

    q = entity_query.lower()

    exact_rows = (
        entity_df
        .filter(
            (F.lower(F.col("entity_name")) == q)
            | (F.lower(F.col("entity_id")) == q)
            | (F.lower(F.col("iso2_code")) == q)
        )
        .collect()
    )

    if len(exact_rows) == 1:
        return json_safe(
            exact_rows[0].asDict(recursive=True)
        )

    if len(exact_rows) > 1:
        candidates = rows_to_dicts(exact_rows)

        raise ValueError(
            "Entity is ambiguous. Exact matching returned multiple "
            f"entities for '{entity_query}': "
            f"{json.dumps(candidates, ensure_ascii=False)}"
        )

    candidates = search_entities(
        entity_query,
        limit=10,
    )

    raise ValueError(
        f"Could not resolve entity '{entity_query}' exactly. "
        "No entity was selected automatically. "
        f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
    )


print("Entity resolution loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Deterministic indicator-series execution

# COMMAND ----------

def get_indicator_series(
    entity_query: str,
    indicator_code: str,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    """
    Return historical observations for one resolved entity and indicator.

    Missing observations are retained as value=None.
    """
    indicator_code = validate_indicator(
        indicator_code
    )

    start_year, end_year = validate_year_range(
        start_year,
        end_year,
    )

    entity = resolve_entity(
        entity_query
    )

    rows = (
        macro_df
        .filter(
            (F.col("entity_id") == entity["entity_id"])
            & (F.col("indicator_id") == indicator_code)
            & F.col("year").between(
                start_year,
                end_year,
            )
        )
        .select(
            "year",
            "value",
            "has_value",
            "unit_type",
            "unit_label",
            "value_type",
            "source_system",
        )
        .orderBy("year")
        .collect()
    )

    observations = rows_to_dicts(rows)

    observed_count = sum(
        1
        for row in observations
        if row["value"] is not None
    )

    missing_count = (
        end_year - start_year + 1
    ) - observed_count

    return {
        "entity": entity,
        "indicator_code": indicator_code,
        "indicator_name": INDICATOR_REGISTRY[
            indicator_code
        ],
        "start_year": start_year,
        "end_year": end_year,
        "observations": observations,
        "observation_count": len(observations),
        "non_null_observation_count": observed_count,
        "missing_year_count": missing_count,
        "historical_only": True,
    }


print("get_indicator_series() loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Data Agent plan validation
# MAGIC
# MAGIC The Data Agent consumes the structured portion of the Supervisor contract created in Notebook 03.

# COMMAND ----------

def validate_structured_plan(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that the Supervisor actually requested structured execution.
    """
    if not isinstance(plan, dict):
        raise TypeError(
            "Supervisor plan must be a dictionary."
        )

    route = plan.get("route")

    if route not in {
        "structured",
        "hybrid",
    }:
        raise ValueError(
            "Data Agent only executes structured or hybrid plans. "
            f"Received route={route!r}."
        )

    if plan.get("needs_structured_tool") is not True:
        raise ValueError(
            "Supervisor plan does not authorize structured-tool execution."
        )

    entities = plan.get(
        "countries_or_entities",
        [],
    ) or []

    indicators = plan.get(
        "indicators",
        [],
    ) or []

    start_year = plan.get(
        "observation_start_year"
    )

    end_year = plan.get(
        "observation_end_year"
    )

    if not entities:
        raise ValueError(
            "Structured execution requires at least one "
            "country/economy/aggregate."
        )

    if len(entities) > MAX_ENTITIES:
        raise ValueError(
            f"Too many entities requested. Maximum is {MAX_ENTITIES}."
        )

    if not indicators:
        raise ValueError(
            "Structured execution requires at least one governed indicator."
        )

    for indicator_code in indicators:
        validate_indicator(
            indicator_code
        )

    if start_year is None:
        raise ValueError(
            "Structured execution requires observation_start_year."
        )

    # "Since YEAR" may arrive from the Supervisor with no explicit end year.
    # Historical tools deterministically close that range at the latest
    # available historical year.
    if end_year is None:
        end_year = MAX_OBSERVATION_YEAR

    start_year, end_year = validate_year_range(
        start_year,
        end_year,
    )

    return {
        "route": route,
        "original_query": plan.get(
            "original_query",
            "",
        ),
        "resolved_query": plan.get(
            "resolved_query",
            plan.get("original_query", ""),
        ),
        "entities": [
            str(entity).strip()
            for entity in entities
            if str(entity).strip()
        ],
        "indicators": list(
            dict.fromkeys(indicators)
        ),
        "start_year": start_year,
        "end_year": end_year,
    }


print("Structured-plan validation loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Data Agent execution
# MAGIC
# MAGIC The agent executes only deterministic governed functions.
# MAGIC It does not ask an LLM to write SQL.

# COMMAND ----------

def run_data_agent(
    supervisor_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the structured portion of a validated Supervisor plan.

    Output is designed for the later Synthesis Agent.
    """
    start = time.perf_counter()

    plan = validate_structured_plan(
        supervisor_plan
    )

    resolved_entities = []
    series_results = []

    # Resolve every entity before running data queries.
    # This avoids partial execution if one entity is ambiguous.
    for entity_query in plan["entities"]:
        resolved = resolve_entity(
            entity_query
        )

        resolved_entities.append({
            "requested_name": entity_query,
            "resolved_entity": resolved,
        })

    for entity_item in resolved_entities:
        entity = entity_item[
            "resolved_entity"
        ]

        # Use the canonical entity name after safe resolution.
        canonical_name = entity[
            "entity_name"
        ]

        for indicator_code in plan[
            "indicators"
        ]:
            result = get_indicator_series(
                entity_query=canonical_name,
                indicator_code=indicator_code,
                start_year=plan["start_year"],
                end_year=plan["end_year"],
            )

            series_results.append(
                result
            )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "agent": "data_agent",
        "status": "success",
        "route": plan["route"],
        "query": plan["resolved_query"],
        "historical_observation_window": {
            "start_year": plan["start_year"],
            "end_year": plan["end_year"],
        },
        "resolved_entities": resolved_entities,
        "indicator_count": len(
            plan["indicators"]
        ),
        "series_count": len(
            series_results
        ),
        "series": series_results,
        "execution_latency_ms": round(
            latency_ms,
            2,
        ),
        "data_contract": {
            "source": MACRO_TABLE,
            "historical_only": True,
            "max_historical_year": MAX_OBSERVATION_YEAR,
            "missing_values_preserved_as_null": True,
            "arbitrary_sql_exposed": False,
        },
    }


print("run_data_agent() loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Compact synthesis payload
# MAGIC
# MAGIC The Synthesis Agent does not need every internal field.
# MAGIC This helper creates a compact factual payload without changing values.

# COMMAND ----------

def build_structured_synthesis_payload(
    data_agent_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a compact structured evidence package for later synthesis.
    """
    if data_agent_result.get(
        "status"
    ) != "success":
        raise ValueError(
            "Cannot build synthesis payload from a failed Data Agent result."
        )

    compact_series = []

    for series in data_agent_result[
        "series"
    ]:
        compact_series.append({
            "entity_id": series[
                "entity"
            ]["entity_id"],
            "entity_name": series[
                "entity"
            ]["entity_name"],
            "entity_type": series[
                "entity"
            ]["entity_type"],
            "indicator_code": series[
                "indicator_code"
            ],
            "indicator_name": series[
                "indicator_name"
            ],
            "start_year": series[
                "start_year"
            ],
            "end_year": series[
                "end_year"
            ],
            "observations": [
                {
                    "year": obs["year"],
                    "value": obs["value"],
                    "unit_label": obs[
                        "unit_label"
                    ],
                }
                for obs in series[
                    "observations"
                ]
            ],
            "missing_year_count": series[
                "missing_year_count"
            ],
        })

    return {
        "evidence_type": "structured_historical",
        "query": data_agent_result[
            "query"
        ],
        "source_table": MACRO_TABLE,
        "historical_only": True,
        "series": compact_series,
    }


print("Structured synthesis payload helper loaded.")

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

print("data_agent_runtime READY")
print(" - run_data_agent")
print(" - build_structured_synthesis_payload")
