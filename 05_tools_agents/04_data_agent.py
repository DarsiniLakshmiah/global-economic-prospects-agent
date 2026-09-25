# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 04_data_agent
# MAGIC
# MAGIC Deterministic structured-data execution layer for the Global Economic Prospects Intelligence Agent.
# MAGIC
# MAGIC This notebook executes the **structured portion** of a validated Supervisor plan against
# MAGIC `worldbank_ai.gold.macroeconomic_indicators`.
# MAGIC
# MAGIC ### Responsibilities
# MAGIC - validate Supervisor plans
# MAGIC - resolve countries, economies, and aggregates safely
# MAGIC - execute governed historical indicator lookups
# MAGIC - compare multiple entities
# MAGIC - preserve missing observations as `NULL`
# MAGIC - reject unsupported/future historical years
# MAGIC - return a compact, JSON-safe result for the Synthesis Agent
# MAGIC
# MAGIC ### Non-responsibilities
# MAGIC - no unrestricted SQL generation
# MAGIC - no GEP document retrieval
# MAGIC - no forecasts
# MAGIC - no final natural-language answer
# MAGIC - no invented values or entity matches

# COMMAND ----------

# MAGIC %md
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
print("Available rows:", macro_df.count())

display(
    macro_df
    .select(
        "entity_id",
        "entity_name",
        "entity_type",
        "indicator_id",
        "indicator_display_name",
        "year",
        "value",
        "unit_label",
        "has_value",
    )
    .orderBy("entity_name", "indicator_id", "year")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
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
# MAGIC ## 10. Test 1 — India GDP growth, 2015–2025
# MAGIC
# MAGIC This mirrors the structured Supervisor route validated in Notebook 03.

# COMMAND ----------

structured_plan = {
    "original_query": (
        "Show India's GDP growth from 2015 to 2025"
    ),
    "resolved_query": (
        "Show India's GDP growth from 2015 to 2025"
    ),
    "route": "structured",
    "countries_or_entities": [
        "India"
    ],
    "indicators": [
        "NY.GDP.MKTP.KD.ZG"
    ],
    "observation_start_year": 2015,
    "observation_end_year": 2025,
    "report_years": [],
    "needs_structured_tool": True,
    "needs_research_tool": False,
}

india_result = run_data_agent(
    structured_plan
)

print(
    json.dumps(
        india_result,
        indent=2,
        ensure_ascii=False,
    )
)

assert india_result[
    "status"
] == "success"

assert india_result[
    "series_count"
] == 1

assert india_result[
    "series"
][0]["entity"]["entity_name"].lower() == "india"

assert india_result[
    "series"
][0]["indicator_code"] == "NY.GDP.MKTP.KD.ZG"

print("India GDP-growth execution test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Verify missing values are preserved
# MAGIC
# MAGIC We validate the behavior against the governed table itself rather than inventing a test value.

# COMMAND ----------

null_row = (
    macro_df
    .filter(
        F.col("value").isNull()
    )
    .select(
        "entity_name",
        "indicator_id",
        "year",
    )
    .orderBy(
        "entity_name",
        "indicator_id",
        "year",
    )
    .limit(1)
    .collect()
)

assert null_row, (
    "Expected at least one NULL historical observation "
    "because the structured pipeline intentionally preserves missing data."
)

null_case = null_row[0]

null_test_result = get_indicator_series(
    entity_query=null_case[
        "entity_name"
    ],
    indicator_code=null_case[
        "indicator_id"
    ],
    start_year=int(
        null_case["year"]
    ),
    end_year=int(
        null_case["year"]
    ),
)

matching_observation = [
    row
    for row in null_test_result[
        "observations"
    ]
    if row["year"] == int(
        null_case["year"]
    )
]

assert matching_observation, (
    "Expected the NULL observation row to remain in the returned series."
)

assert matching_observation[
    0
]["value"] is None

print(
    "NULL preservation test passed:",
    null_case["entity_name"],
    null_case["indicator_id"],
    null_case["year"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Test 2 — hybrid structured portion
# MAGIC
# MAGIC Only the historical-data portion is executed here.
# MAGIC The Research Agent will handle the GEP outlook portion in Notebook 05.

# COMMAND ----------

def find_exact_entity_name(
    candidates: List[str],
) -> Optional[str]:
    """
    Return the first candidate that exists as an exact governed entity name.
    No fuzzy selection.
    """
    names_df = (
        macro_df
        .select("entity_name")
        .dropDuplicates()
    )

    for candidate in candidates:
        exists = (
            names_df
            .filter(
                F.lower(
                    F.col("entity_name")
                )
                == candidate.lower()
            )
            .limit(1)
            .count()
            == 1
        )

        if exists:
            return candidate

    return None


south_asia_name = find_exact_entity_name(
    ["South Asia"]
)

ssa_name = find_exact_entity_name(
    [
        "Sub-Saharan Africa",
        "Sub-Saharan Africa (excluding high income)",
    ]
)

print(
    "South Asia exact entity:",
    south_asia_name,
)

print(
    "Sub-Saharan Africa exact entity:",
    ssa_name,
)

if (
    south_asia_name is not None
    and ssa_name is not None
):
    hybrid_plan = {
        "original_query": (
            "Compare GDP growth in South Asia and Sub-Saharan Africa "
            "since 2022 and explain the World Bank outlook for the two regions."
        ),
        "resolved_query": (
            "Compare GDP growth in South Asia and Sub-Saharan Africa "
            "since 2022 and explain the World Bank outlook for the two regions."
        ),
        "route": "hybrid",
        "countries_or_entities": [
            south_asia_name,
            ssa_name,
        ],
        "indicators": [
            "NY.GDP.MKTP.KD.ZG"
        ],
        "observation_start_year": 2022,
        "observation_end_year": None,
        "report_years": [],
        "needs_structured_tool": True,
        "needs_research_tool": True,
    }

    hybrid_structured_result = (
        run_data_agent(
            hybrid_plan
        )
    )

    print(
        json.dumps(
            build_structured_synthesis_payload(
                hybrid_structured_result
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    assert hybrid_structured_result[
        "series_count"
    ] == 2

    assert hybrid_structured_result[
        "historical_observation_window"
    ]["end_year"] == MAX_OBSERVATION_YEAR

    print(
        "Hybrid structured-data execution test passed."
    )

else:
    print(
        "Hybrid auto-test skipped because one or both requested "
        "aggregate names were not present as exact governed entity names."
    )
    print(
        "No fuzzy entity was silently selected."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Guardrail — reject future historical observations

# COMMAND ----------

future_guardrail_passed = False

try:
    run_data_agent({
        "original_query": "Show India's GDP growth through 2026",
        "resolved_query": "Show India's GDP growth through 2026",
        "route": "structured",
        "countries_or_entities": ["India"],
        "indicators": ["NY.GDP.MKTP.KD.ZG"],
        "observation_start_year": 2022,
        "observation_end_year": 2026,
        "needs_structured_tool": True,
        "needs_research_tool": False,
    })

except ValueError as exc:
    future_guardrail_passed = True
    print(
        "Expected historical-year rejection:"
    )
    print(str(exc))

assert future_guardrail_passed

print("Future historical-observation guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Guardrail — reject a pure RAG plan

# COMMAND ----------

rag_rejection_passed = False

try:
    run_data_agent({
        "original_query": (
            "What risks did the World Bank identify for South Asia in 2025?"
        ),
        "resolved_query": (
            "What risks did the World Bank identify for South Asia in 2025?"
        ),
        "route": "rag",
        "countries_or_entities": ["South Asia"],
        "indicators": [],
        "observation_start_year": None,
        "observation_end_year": None,
        "report_years": [2025],
        "needs_structured_tool": False,
        "needs_research_tool": True,
    })

except ValueError as exc:
    rag_rejection_passed = True
    print("Expected route rejection:")
    print(str(exc))

assert rag_rejection_passed

print("Pure-RAG route rejection passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Guardrail — unsupported indicator

# COMMAND ----------

indicator_guardrail_passed = False

try:
    run_data_agent({
        "original_query": "Show unemployment in India",
        "resolved_query": "Show unemployment in India",
        "route": "structured",
        "countries_or_entities": ["India"],
        "indicators": ["SL.UEM.TOTL.ZS"],
        "observation_start_year": 2020,
        "observation_end_year": 2025,
        "needs_structured_tool": True,
        "needs_research_tool": False,
    })

except ValueError as exc:
    indicator_guardrail_passed = True
    print("Expected indicator rejection:")
    print(str(exc))

assert indicator_guardrail_passed

print("Unsupported-indicator guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Guardrail — ambiguous/unresolved entity
# MAGIC
# MAGIC We deliberately use a non-existent name and verify that the agent does not invent a match.

# COMMAND ----------

entity_guardrail_passed = False

try:
    resolve_entity(
        "Definitely Not A World Bank Entity XYZ"
    )

except ValueError as exc:
    entity_guardrail_passed = True
    print("Expected entity-resolution rejection:")
    print(str(exc))

assert entity_guardrail_passed

print("Entity-resolution guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Data Agent callable schema
# MAGIC
# MAGIC This schema is for orchestration/serving later. The Supervisor plan remains the source of routing intent.

# COMMAND ----------

DATA_AGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_data_agent",
        "description": (
            "Execute the structured historical-data portion of a validated "
            "Supervisor plan using governed World Bank Gold data. "
            "Does not generate SQL, forecasts, GEP research, or a final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "supervisor_plan": {
                    "type": "object",
                    "description": (
                        "Validated Supervisor plan with route, entities, "
                        "indicators, and historical observation years."
                    ),
                }
            },
            "required": [
                "supervisor_plan"
            ],
            "additionalProperties": False,
        },
    },
}

print(
    json.dumps(
        DATA_AGENT_TOOL_SCHEMA,
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Final validation suite

# COMMAND ----------

validation_results = []


def record_test(
    name: str,
    passed: bool,
    detail: str,
):
    validation_results.append({
        "test": name,
        "passed": bool(passed),
        "detail": detail,
    })


record_test(
    "gold_schema_contract",
    not missing_columns,
    (
        "Physical indicator_id / indicator_display_name "
        "contract validated"
    ),
)

record_test(
    "india_structured_execution",
    (
        india_result["status"] == "success"
        and india_result["series_count"] == 1
    ),
    "India GDP growth 2015-2025 executed",
)

record_test(
    "null_preservation",
    matching_observation[0]["value"] is None,
    "Missing historical observation remained NULL",
)

record_test(
    "future_year_guardrail",
    future_guardrail_passed,
    "2026 rejected from historical observation path",
)

record_test(
    "rag_route_guardrail",
    rag_rejection_passed,
    "Pure RAG plan rejected by Data Agent",
)

record_test(
    "indicator_allowlist",
    indicator_guardrail_passed,
    "Unsupported indicator rejected",
)

record_test(
    "entity_resolution_safety",
    entity_guardrail_passed,
    "Unresolved entity was not silently guessed",
)

record_test(
    "no_arbitrary_sql",
    True,
    (
        "Data Agent exposes deterministic functions only; "
        "no SQL string execution interface exists"
    ),
)

payload = build_structured_synthesis_payload(
    india_result
)

record_test(
    "synthesis_payload",
    (
        payload["evidence_type"]
        == "structured_historical"
        and len(payload["series"]) == 1
    ),
    "Compact structured evidence payload built",
)

validation_df = spark.createDataFrame(
    validation_results
)

display(validation_df)

failed_tests = [
    item["test"]
    for item in validation_results
    if not item["passed"]
]

assert not failed_tests, (
    f"Data Agent validation failures: {failed_tests}"
)

print("All Data Agent validation tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Final status

# COMMAND ----------

print("")
print("04_data_agent COMPLETE")
print("")
print("Validated path:")
print(
    "Supervisor structured plan -> validation -> safe entity resolution -> "
    "governed Gold indicator lookup -> NULL-preserving historical series -> "
    "structured synthesis payload"
)
print("")
print("Important:")
print(" - No unrestricted text-to-SQL")
print(" - No arbitrary SQL execution tool")
print(" - Physical Gold indicator column is indicator_id")
print(" - Agent-facing field remains indicator_code")
print(" - Missing observations remain NULL")
print(" - Historical observations stop at 2025")
print(" - Forecasts remain on the GEP research path")
print(" - Entity candidates are never silently selected")
print("")
print("Next notebook after this passes:")
print("05_tools_agents / 05_research_agent")