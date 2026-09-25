# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 01_structured_data_tools
# MAGIC
# MAGIC Governed structured-data tool layer for the **Global Economic Prospects Intelligence Agent**.
# MAGIC
# MAGIC This notebook intentionally does **not** expose arbitrary SQL to the future agent.
# MAGIC It provides deterministic, validated tools over the Gold macroeconomic table.
# MAGIC
# MAGIC ### Tools created
# MAGIC - `list_available_indicators`
# MAGIC - `search_entities`
# MAGIC - `resolve_entity`
# MAGIC - `get_indicator_series`
# MAGIC - `compare_indicator_series`
# MAGIC - `get_latest_indicator`
# MAGIC - `execute_structured_tool`
# MAGIC
# MAGIC ### Guardrails
# MAGIC - Historical observations are kept separate from GEP forecasts.
# MAGIC - Missing values remain missing; they are never converted to zero.
# MAGIC - Indicator codes come from a governed allow-list.
# MAGIC - Entity names/IDs are resolved against governed Gold data.
# MAGIC - Year ranges and result sizes are bounded.
# MAGIC - No unrestricted `execute_sql()` tool is exposed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Configuration

# COMMAND ----------

from typing import Any, Dict, List
from decimal import Decimal
import json

from pyspark.sql import functions as F

CATALOG = "worldbank_ai"
GOLD_SCHEMA = "gold"
AI_SCHEMA = "ai"
MONITORING_SCHEMA = "monitoring"

MACRO_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.macroeconomic_indicators"
COUNTRY_SUMMARY_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.country_summary"

# The structured ingestion pipeline currently contains historical observations
# from 2010 through 2025. 2026 GEP forecasts belong to the document/forecast
# path and must not be silently mixed into this historical tool.
MIN_OBSERVATION_YEAR = 2010
MAX_OBSERVATION_YEAR = 2025

MAX_YEAR_SPAN = 30
MAX_ENTITIES = 20

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{AI_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}")

print("Structured tool configuration loaded.")
print("Macro table:", MACRO_TABLE)
print("Country summary:", COUNTRY_SUMMARY_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Preflight — validate the actual Gold contract
# MAGIC
# MAGIC We fail early if the Gold table does not match the actual governed contract.
# MAGIC
# MAGIC Physical Gold mapping used by this notebook:
# MAGIC - `indicator_id` = World Bank indicator code
# MAGIC - `indicator_display_name` = human-readable indicator name
# MAGIC
# MAGIC The public tool argument remains `indicator_code` because that is clearer
# MAGIC for an LLM/tool-calling interface. Internally it filters `indicator_id`.

# COMMAND ----------

required_tables = [
    MACRO_TABLE,
    COUNTRY_SUMMARY_TABLE,
]

for table_name in required_tables:
    if not spark.catalog.tableExists(table_name):
        raise ValueError(
            f"Required Gold table does not exist: {table_name}. "
            "Run/validate the structured Gold notebook before this notebook."
        )

macro_df = spark.table(MACRO_TABLE)

REQUIRED_MACRO_COLUMNS = {
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

missing_macro_columns = sorted(
    REQUIRED_MACRO_COLUMNS - set(macro_df.columns)
)

if missing_macro_columns:
    print("Actual macro table columns:")
    print(macro_df.columns)
    raise ValueError(
        "Gold macro table does not match the structured-tool contract. "
        f"Missing columns: {missing_macro_columns}"
    )

print(f"{MACRO_TABLE}: {macro_df.count():,} rows")
print(f"{COUNTRY_SUMMARY_TABLE}: {spark.table(COUNTRY_SUMMARY_TABLE).count():,} rows")
print("Gold table contract validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Inspect the Gold data

# COMMAND ----------

# ============================================================
# 03. Inspect the actual Gold data
# ============================================================

macro_df.printSchema()

display(
    macro_df
    .select(
        "entity_id",
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
    )
    .orderBy(
        "entity_name",
        "indicator_id",
        "year",
    )
    .limit(20)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Governed indicator registry
# MAGIC
# MAGIC The agent may only request indicators in this registry.

# COMMAND ----------

INDICATOR_REGISTRY = {
    "NY.GDP.MKTP.KD.ZG": {
        "name": "GDP growth",
        "unit": "annual %",
        "category": "growth",
    },
    "NY.GDP.PCAP.KD.ZG": {
        "name": "GDP per capita growth",
        "unit": "annual %",
        "category": "growth",
    },
    "NY.GDP.MKTP.CD": {
        "name": "GDP",
        "unit": "current US$",
        "category": "output",
    },
    "NY.GDP.PCAP.CD": {
        "name": "GDP per capita",
        "unit": "current US$",
        "category": "output",
    },
    "FP.CPI.TOTL.ZG": {
        "name": "Inflation, consumer prices",
        "unit": "annual %",
        "category": "inflation",
    },
    "NE.TRD.GNFS.ZS": {
        "name": "Trade",
        "unit": "% of GDP",
        "category": "trade",
    },
    "NE.EXP.GNFS.ZS": {
        "name": "Exports of goods and services",
        "unit": "% of GDP",
        "category": "trade",
    },
    "NE.IMP.GNFS.ZS": {
        "name": "Imports of goods and services",
        "unit": "% of GDP",
        "category": "trade",
    },
    "NE.GDI.TOTL.ZS": {
        "name": "Gross capital formation",
        "unit": "% of GDP",
        "category": "investment",
    },
    "GC.XPN.TOTL.GD.ZS": {
        "name": "Expense",
        "unit": "% of GDP",
        "category": "fiscal",
    },
    "GC.REV.XGRT.GD.ZS": {
        "name": "Revenue excluding grants",
        "unit": "% of GDP",
        "category": "fiscal",
    },
    "GC.DOD.TOTL.GD.ZS": {
        "name": "Central government debt",
        "unit": "% of GDP",
        "category": "fiscal",
    },
    "NY.GDS.TOTL.ZS": {
        "name": "Gross domestic savings",
        "unit": "% of GDP",
        "category": "savings",
    },
    "BX.KLT.DINV.WD.GD.ZS": {
        "name": "Foreign direct investment, net inflows",
        "unit": "% of GDP",
        "category": "investment",
    },
    "BN.CAB.XOKA.GD.ZS": {
        "name": "Current account balance",
        "unit": "% of GDP",
        "category": "external",
    },
}

assert len(INDICATOR_REGISTRY) == 15
print("Governed indicators:", len(INDICATOR_REGISTRY))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Validation helpers

# COMMAND ----------

def validate_indicator(indicator_code: str) -> str:
    """Validate and canonicalize a governed indicator code."""
    if indicator_code is None:
        raise ValueError("indicator_code cannot be None.")

    indicator_code = str(indicator_code).strip().upper()

    if indicator_code not in INDICATOR_REGISTRY:
        raise ValueError(
            f"Unsupported indicator: {indicator_code}. "
            "Call list_available_indicators() to inspect the governed set."
        )

    return indicator_code


def validate_year_range(start_year: int, end_year: int):
    """Validate a historical observation year range."""
    if start_year is None or end_year is None:
        raise ValueError("start_year and end_year are required.")

    start_year = int(start_year)
    end_year = int(end_year)

    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year.")

    if start_year < MIN_OBSERVATION_YEAR:
        raise ValueError(
            f"start_year must be >= {MIN_OBSERVATION_YEAR}."
        )

    if end_year > MAX_OBSERVATION_YEAR:
        raise ValueError(
            f"Historical observations currently end at {MAX_OBSERVATION_YEAR}. "
            "Do not treat GEP forecasts as historical observations."
        )

    if (end_year - start_year + 1) > MAX_YEAR_SPAN:
        raise ValueError(
            f"Requested range exceeds the {MAX_YEAR_SPAN}-year tool limit."
        )

    return start_year, end_year


def json_safe(value: Any):
    """Convert returned values into JSON-safe values without changing nulls."""
    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    return value


def tool_result_to_json(result: Dict[str, Any]) -> str:
    """Serialize a tool result for an LLM/tool-calling boundary."""
    return json.dumps(
        json_safe(result),
        ensure_ascii=False,
        indent=2,
    )

print("Validation and serialization helpers loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Tool — list available indicators

# COMMAND ----------

def list_available_indicators() -> List[Dict[str, Any]]:
    """Return the indicators available to the structured-data agent."""
    return [
        {
            "indicator_code": code,
            "indicator_name": metadata["name"],
            "unit": metadata["unit"],
            "category": metadata["category"],
        }
        for code, metadata in INDICATOR_REGISTRY.items()
    ]


available_indicators = list_available_indicators()
assert len(available_indicators) == 15

display(
    spark.createDataFrame(available_indicators)
    .orderBy("category", "indicator_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Tool — search governed entities

# COMMAND ----------

def search_entities(
    search_text: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search governed World Bank countries/economies/aggregates by
    entity name or canonical entity ID.
    """
    if not search_text or not str(search_text).strip():
        raise ValueError("search_text cannot be empty.")

    search_text = str(search_text).strip().lower()
    limit = min(max(int(limit), 1), 25)

    entity_df = (
        spark.table(MACRO_TABLE)
        .select(
            "entity_id",
            "entity_name",
            "entity_type",
        )
        .distinct()
        .filter(
            F.lower(F.col("entity_name")).contains(search_text)
            | F.lower(F.col("entity_id")).contains(search_text)
        )
        .orderBy("entity_name")
        .limit(limit)
    )

    return [
        row.asDict(recursive=True)
        for row in entity_df.collect()
    ]


print("India search:")
print(json.dumps(search_entities("India"), indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Exact entity resolver
# MAGIC
# MAGIC The resolver never silently chooses among fuzzy matches.

# COMMAND ----------

def resolve_entity(entity: str) -> Dict[str, Any]:
    """
    Resolve an exact entity name or canonical ID.

    If there is no exact match, return candidate names in the error
    rather than silently choosing one.
    """
    if not entity or not str(entity).strip():
        raise ValueError("entity cannot be empty.")

    entity = str(entity).strip()

    df = (
        spark.table(MACRO_TABLE)
        .select(
            "entity_id",
            "entity_name",
            "entity_type",
        )
        .distinct()
    )

    exact_matches = (
        df
        .filter(
            (F.lower(F.col("entity_id")) == entity.lower())
            | (F.lower(F.col("entity_name")) == entity.lower())
        )
        .limit(2)
        .collect()
    )

    if len(exact_matches) == 1:
        return exact_matches[0].asDict(recursive=True)

    if len(exact_matches) > 1:
        raise ValueError(
            f"Entity '{entity}' is ambiguous in the governed entity dimension."
        )

    candidates = search_entities(entity, limit=5)

    if not candidates:
        raise ValueError(
            f"No governed entity found for '{entity}'."
        )

    candidate_names = [
        f"{item['entity_name']} ({item['entity_id']})"
        for item in candidates
    ]

    raise ValueError(
        f"No exact entity match for '{entity}'. "
        f"Possible matches: {candidate_names}"
    )


india_entity = resolve_entity("India")
print("Resolved India:")
print(json.dumps(india_entity, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Tool — historical indicator series

# COMMAND ----------

def get_indicator_series(
    entity: str,
    indicator_code: str,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    """
    Retrieve one governed historical World Bank indicator series.

    Missing observations are preserved as null.
    GEP forecasts are intentionally excluded.
    """
    resolved_entity = resolve_entity(entity)
    indicator_code = validate_indicator(indicator_code)
    start_year, end_year = validate_year_range(start_year, end_year)

    result_df = (
        spark.table(MACRO_TABLE)
        .filter(
            (F.col("entity_id") == resolved_entity["entity_id"])
            & (F.col("indicator_id") == indicator_code)
            & (F.col("year").between(start_year, end_year))
        )
        .select(
            "entity_id",
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
        )
        .orderBy("year")
    )

    rows = [
        row.asDict(recursive=True)
        for row in result_df.collect()
    ]

    return json_safe({
        "tool": "get_indicator_series",
        "status": "OK",
        "entity": resolved_entity,
        "indicator": {
            "indicator_code": indicator_code,
            **INDICATOR_REGISTRY[indicator_code],
        },
        "start_year": start_year,
        "end_year": end_year,
        "observation_type": "historical",
        "data": rows,
        "observation_count": len(rows),
        "non_null_observation_count": sum(
            row["value"] is not None for row in rows
        ),
    })


india_gdp_growth = get_indicator_series(
    entity="India",
    indicator_code="NY.GDP.MKTP.KD.ZG",
    start_year=2015,
    end_year=2025,
)

print(tool_result_to_json(india_gdp_growth))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Validate the historical-series tool

# COMMAND ----------

assert india_gdp_growth["entity"]["entity_name"] == "India"
assert india_gdp_growth["indicator"]["indicator_code"] == "NY.GDP.MKTP.KD.ZG"
assert india_gdp_growth["start_year"] == 2015
assert india_gdp_growth["end_year"] == 2025
assert india_gdp_growth["observation_type"] == "historical"

print("Historical indicator smoke test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Tool — compare multiple entities

# COMMAND ----------

def compare_indicator_series(
    entities: List[str],
    indicator_code: str,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    """
    Compare the same governed historical indicator across multiple entities.
    """
    if not entities:
        raise ValueError("At least one entity is required.")

    if len(entities) > MAX_ENTITIES:
        raise ValueError(
            f"A maximum of {MAX_ENTITIES} entities can be compared in one call."
        )

    indicator_code = validate_indicator(indicator_code)
    start_year, end_year = validate_year_range(start_year, end_year)

    resolved_entities = [
        resolve_entity(entity)
        for entity in entities
    ]

    entity_ids = [
        item["entity_id"]
        for item in resolved_entities
    ]

    result_df = (
        spark.table(MACRO_TABLE)
        .filter(
            F.col("entity_id").isin(entity_ids)
            & (F.col("indicator_id") == indicator_code)
            & (F.col("year").between(start_year, end_year))
        )
        .select(
            "entity_id",
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
        )
        .orderBy("year", "entity_name")
    )

    rows = [
        row.asDict(recursive=True)
        for row in result_df.collect()
    ]

    return json_safe({
        "tool": "compare_indicator_series",
        "status": "OK",
        "entities": resolved_entities,
        "indicator": {
            "indicator_code": indicator_code,
            **INDICATOR_REGISTRY[indicator_code],
        },
        "start_year": start_year,
        "end_year": end_year,
        "observation_type": "historical",
        "data": rows,
        "observation_count": len(rows),
    })

print("Multi-entity comparison tool loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validate regional aggregate names before comparison
# MAGIC
# MAGIC We search the governed data first instead of guessing World Bank aggregate names.

# COMMAND ----------

south_asia_matches = search_entities("South Asia")
ssa_matches = search_entities("Sub-Saharan Africa")

print("South Asia candidates:")
print(json.dumps(south_asia_matches, indent=2, default=str))

print("\nSub-Saharan Africa candidates:")
print(json.dumps(ssa_matches, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Regional comparison smoke test
# MAGIC
# MAGIC This runs only when the exact canonical names below exist in the Gold table.
# MAGIC If the World Bank aggregate has a qualified name such as "(excluding high income)",
# MAGIC the search output above shows the exact value to use.

# COMMAND ----------

def exact_entity_exists(name: str) -> bool:
    return (
        spark.table(MACRO_TABLE)
        .select("entity_name")
        .distinct()
        .filter(F.lower(F.col("entity_name")) == name.lower())
        .limit(1)
        .count()
        == 1
    )


if exact_entity_exists("South Asia") and exact_entity_exists("Sub-Saharan Africa"):
    regional_growth = compare_indicator_series(
        entities=[
            "South Asia",
            "Sub-Saharan Africa",
        ],
        indicator_code="NY.GDP.MKTP.KD.ZG",
        start_year=2022,
        end_year=2025,
    )

    print(tool_result_to_json(regional_growth))
    print("Regional comparison smoke test passed.")
else:
    print(
        "Regional comparison was not auto-run because one or both exact "
        "aggregate names differ in the governed Gold data."
    )
    print("Use the exact names printed by the previous cell.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Tool — latest available historical observation

# COMMAND ----------

def get_latest_indicator(
    entity: str,
    indicator_code: str,
) -> Dict[str, Any]:
    """
    Return the latest available NON-NULL historical observation.

    Latest available is not assumed to be 2025 because some indicators
    have reporting lags.
    """
    resolved_entity = resolve_entity(entity)
    indicator_code = validate_indicator(indicator_code)

    rows = (
        spark.table(MACRO_TABLE)
        .filter(
            (F.col("entity_id") == resolved_entity["entity_id"])
            & (F.col("indicator_id") == indicator_code)
            & F.col("has_value")
            & F.col("value").isNotNull()
        )
        .select(
            "entity_id",
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
        )
        .orderBy(F.desc("year"))
        .limit(1)
        .collect()
    )

    if not rows:
        return {
            "tool": "get_latest_indicator",
            "status": "NO_AVAILABLE_OBSERVATION",
            "entity": resolved_entity,
            "indicator": {
                "indicator_code": indicator_code,
                **INDICATOR_REGISTRY[indicator_code],
            },
            "observation_type": "historical",
            "data": None,
        }

    return json_safe({
        "tool": "get_latest_indicator",
        "status": "OK",
        "entity": resolved_entity,
        "indicator": {
            "indicator_code": indicator_code,
            **INDICATOR_REGISTRY[indicator_code],
        },
        "observation_type": "historical",
        "data": rows[0].asDict(recursive=True),
    })


latest_india_growth = get_latest_indicator(
    entity="India",
    indicator_code="NY.GDP.MKTP.KD.ZG",
)

print(tool_result_to_json(latest_india_growth))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Validate missing-value behavior
# MAGIC
# MAGIC Missing numeric observations must remain `None`/SQL `NULL`.
# MAGIC We never convert them to zero.

# COMMAND ----------

null_count = (
    spark.table(MACRO_TABLE)
    .filter(F.col("value").isNull())
    .count()
)

assert null_count > 0, (
    "Expected the governed historical table to preserve missing values, "
    "but no NULL observations were found."
)

print(f"Historical NULL observations preserved in Gold: {null_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Validate historical / forecast separation

# COMMAND ----------

forecast_guardrail_passed = False

try:
    get_indicator_series(
        entity="India",
        indicator_code="NY.GDP.MKTP.KD.ZG",
        start_year=2024,
        end_year=2026,
    )

except ValueError as exc:
    forecast_guardrail_passed = True
    print("Expected validation:")
    print(str(exc))

assert forecast_guardrail_passed, (
    "2026 historical request should have been rejected."
)

print("Historical/forecast separation guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Agent-safe tool registry

# COMMAND ----------

STRUCTURED_TOOL_REGISTRY = {
    "list_available_indicators": {
        "function": list_available_indicators,
        "description": (
            "List the governed historical macroeconomic indicators "
            "available to the structured-data agent."
        ),
    },
    "search_entities": {
        "function": search_entities,
        "description": (
            "Search governed World Bank countries, economies, regions, "
            "and aggregate entities."
        ),
    },
    "get_indicator_series": {
        "function": get_indicator_series,
        "description": (
            "Retrieve one historical indicator for one governed entity "
            "over a validated year range."
        ),
    },
    "compare_indicator_series": {
        "function": compare_indicator_series,
        "description": (
            "Compare one historical indicator across multiple governed entities."
        ),
    },
    "get_latest_indicator": {
        "function": get_latest_indicator,
        "description": (
            "Retrieve the latest available non-null historical observation "
            "for one entity and indicator."
        ),
    },
}

print("Registered structured tools:")
for tool_name in STRUCTURED_TOOL_REGISTRY:
    print(" -", tool_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. LLM-facing function schemas
# MAGIC
# MAGIC These schemas constrain the future Data Agent's tool calls.

# COMMAND ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_indicators",
            "description": (
                "List the governed historical macroeconomic indicators "
                "available to the Data Agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": (
                "Search governed World Bank countries, economies, regions, "
                "and aggregate entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_text": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["search_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicator_series",
            "description": (
                "Retrieve historical World Bank observations for one entity, "
                "one indicator, and a year range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "indicator_code": {
                        "type": "string",
                        "enum": list(INDICATOR_REGISTRY.keys()),
                    },
                    "start_year": {
                        "type": "integer",
                        "minimum": MIN_OBSERVATION_YEAR,
                        "maximum": MAX_OBSERVATION_YEAR,
                    },
                    "end_year": {
                        "type": "integer",
                        "minimum": MIN_OBSERVATION_YEAR,
                        "maximum": MAX_OBSERVATION_YEAR,
                    },
                },
                "required": [
                    "entity",
                    "indicator_code",
                    "start_year",
                    "end_year",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_indicator_series",
            "description": (
                "Compare the same historical macroeconomic indicator "
                "across multiple governed entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_ENTITIES,
                    },
                    "indicator_code": {
                        "type": "string",
                        "enum": list(INDICATOR_REGISTRY.keys()),
                    },
                    "start_year": {
                        "type": "integer",
                        "minimum": MIN_OBSERVATION_YEAR,
                        "maximum": MAX_OBSERVATION_YEAR,
                    },
                    "end_year": {
                        "type": "integer",
                        "minimum": MIN_OBSERVATION_YEAR,
                        "maximum": MAX_OBSERVATION_YEAR,
                    },
                },
                "required": [
                    "entities",
                    "indicator_code",
                    "start_year",
                    "end_year",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_indicator",
            "description": (
                "Return the latest available non-null historical observation "
                "for an entity and indicator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "indicator_code": {
                        "type": "string",
                        "enum": list(INDICATOR_REGISTRY.keys()),
                    },
                },
                "required": [
                    "entity",
                    "indicator_code",
                ],
                "additionalProperties": False,
            },
        },
    },
]

assert len(TOOL_SCHEMAS) == len(STRUCTURED_TOOL_REGISTRY)

print(f"Created {len(TOOL_SCHEMAS)} LLM-facing tool schemas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Controlled tool executor
# MAGIC
# MAGIC The future agent calls this dispatcher. It cannot execute an unregistered
# MAGIC function or arbitrary SQL.

# COMMAND ----------

def execute_structured_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute only a registered structured-data tool."""
    if tool_name not in STRUCTURED_TOOL_REGISTRY:
        raise ValueError(
            f"Tool '{tool_name}' is not registered."
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be a dictionary."
        )

    function = STRUCTURED_TOOL_REGISTRY[tool_name]["function"]
    return function(**arguments)


tool_result = execute_structured_tool(
    tool_name="get_indicator_series",
    arguments={
        "entity": "India",
        "indicator_code": "NY.GDP.MKTP.KD.ZG",
        "start_year": 2015,
        "end_year": 2025,
    },
)

print(tool_result_to_json(tool_result))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Security test — reject unregistered execution

# COMMAND ----------

unregistered_tool_rejected = False

try:
    execute_structured_tool(
        tool_name="execute_arbitrary_sql",
        arguments={
            "sql": "DROP TABLE anything"
        },
    )

except ValueError as exc:
    unregistered_tool_rejected = True
    print("Expected rejection:")
    print(str(exc))

assert unregistered_tool_rejected
print("Unregistered-tool guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Functional validation suite

# COMMAND ----------

validation_results = []

def record_test(name: str, passed: bool, detail: str):
    validation_results.append({
        "test": name,
        "passed": bool(passed),
        "detail": detail,
    })


# 1. Registry size
record_test(
    "indicator_registry",
    len(INDICATOR_REGISTRY) == 15,
    f"{len(INDICATOR_REGISTRY)} governed indicators",
)

# 2. India resolves deterministically
record_test(
    "entity_resolution",
    india_entity["entity_name"] == "India",
    f"Resolved to {india_entity}",
)

# 3. Historical series contract
record_test(
    "historical_series",
    (
        india_gdp_growth["indicator"]["indicator_code"]
        == "NY.GDP.MKTP.KD.ZG"
        and india_gdp_growth["start_year"] == 2015
        and india_gdp_growth["end_year"] == 2025
    ),
    (
        f"{india_gdp_growth['observation_count']} returned rows; "
        f"{india_gdp_growth['non_null_observation_count']} non-null"
    ),
)

# 4. Missing values preserved
record_test(
    "null_preservation",
    null_count > 0,
    f"{null_count:,} NULL observations remain NULL",
)

# 5. Forecast/historical separation
record_test(
    "forecast_guardrail",
    forecast_guardrail_passed,
    "2026 rejected by historical-observation tool",
)

# 6. Arbitrary tool execution rejected
record_test(
    "tool_allowlist",
    unregistered_tool_rejected,
    "Unregistered tool call rejected",
)

validation_df = spark.createDataFrame(validation_results)

display(validation_df)

failed_tests = [
    row["test"]
    for row in validation_results
    if not row["passed"]
]

assert not failed_tests, f"Validation failures: {failed_tests}"

print("All structured-tool validation tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Example agent payload
# MAGIC
# MAGIC This is the exact kind of deterministic payload the future Data Agent
# MAGIC will receive after a successful tool call.

# COMMAND ----------

example_agent_payload = execute_structured_tool(
    tool_name="get_indicator_series",
    arguments={
        "entity": "India",
        "indicator_code": "NY.GDP.MKTP.KD.ZG",
        "start_year": 2015,
        "end_year": 2025,
    },
)

print(tool_result_to_json(example_agent_payload))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. Final status

# COMMAND ----------

print("")
print("01_structured_data_tools COMPLETE")
print("")
print("Validated path:")
print(
    "agent-safe arguments -> validation -> governed Gold Delta "
    "-> deterministic result -> JSON-safe tool response"
)
print("")
print("Registered tools:")
for name in STRUCTURED_TOOL_REGISTRY:
    print(" -", name)
print("")
print("Important:")
print(" - No arbitrary SQL tool exposed")
print(" - Missing observations remain NULL")
print(" - Historical observations stop at 2025")
print(" - GEP forecasts remain a separate future tool path")
print("")
print("Next notebook after this passes:")
print("05_tools_agents / 02_research_rag_tool")