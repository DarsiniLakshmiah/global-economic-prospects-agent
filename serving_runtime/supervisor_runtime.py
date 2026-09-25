# Databricks notebook source
# MAGIC %md
# MAGIC # runtime / supervisor_runtime
# MAGIC
# MAGIC Definitions only. No package installation, Python restart, smoke test,
# MAGIC benchmark execution, or development validation cells.
# MAGIC
# MAGIC This notebook is safe to import with `%run` from evaluation/serving notebooks.

# COMMAND ----------

# MAGIC ## 02. Imports and configuration

# COMMAND ----------

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError
import json
import re
import time

from databricks_openai import DatabricksOpenAI

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

VALID_REPORT_YEARS = {2022, 2023, 2024, 2025, 2026}
MIN_OBSERVATION_YEAR = 2010
MAX_OBSERVATION_YEAR = 2025

VALID_ROUTES = {
    "structured",
    "rag",
    "hybrid",
    "temporal_rag",
    "unknown",
}

# Same governed indicator set exposed by Notebook 01.
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

llm_client = DatabricksOpenAI()

print("Supervisor configuration loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Supervisor output contract
# MAGIC
# MAGIC The LLM must return this schema. We validate it before downstream execution.

# COMMAND ----------

class SupervisorPlan(BaseModel):
    original_query: str
    resolved_query: str

    route: Literal[
        "structured",
        "rag",
        "hybrid",
        "temporal_rag",
        "unknown",
    ]

    countries_or_entities: List[str] = Field(default_factory=list)
    indicators: List[str] = Field(default_factory=list)

    observation_start_year: Optional[int] = None
    observation_end_year: Optional[int] = None

    report_years: List[int] = Field(default_factory=list)

    regions: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)

    is_temporal: bool = False
    is_followup: bool = False

    needs_structured_tool: bool = False
    needs_research_tool: bool = False

    reason: str


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Pydantic v1/v2 compatible dictionary conversion."""
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


print("SupervisorPlan contract loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Deterministic indicator detection
# MAGIC
# MAGIC We use deterministic matching before asking the LLM. This reduces unnecessary model work
# MAGIC and keeps known indicator mappings stable.

# COMMAND ----------

INDICATOR_ALIASES = {
    "gdp growth": "NY.GDP.MKTP.KD.ZG",
    "economic growth": "NY.GDP.MKTP.KD.ZG",
    "gdp per capita growth": "NY.GDP.PCAP.KD.ZG",
    "gdp per capita": "NY.GDP.PCAP.CD",
    "inflation": "FP.CPI.TOTL.ZG",
    "consumer prices": "FP.CPI.TOTL.ZG",
    "trade": "NE.TRD.GNFS.ZS",
    "exports": "NE.EXP.GNFS.ZS",
    "imports": "NE.IMP.GNFS.ZS",
    "gross capital formation": "NE.GDI.TOTL.ZS",
    "investment": "NE.GDI.TOTL.ZS",
    "government expense": "GC.XPN.TOTL.GD.ZS",
    "government spending": "GC.XPN.TOTL.GD.ZS",
    "government revenue": "GC.REV.XGRT.GD.ZS",
    "government debt": "GC.DOD.TOTL.GD.ZS",
    "debt": "GC.DOD.TOTL.GD.ZS",
    "domestic savings": "NY.GDS.TOTL.ZS",
    "savings": "NY.GDS.TOTL.ZS",
    "foreign direct investment": "BX.KLT.DINV.WD.GD.ZS",
    "fdi": "BX.KLT.DINV.WD.GD.ZS",
    "current account": "BN.CAB.XOKA.GD.ZS",
}


def detect_indicator_codes(query: str) -> List[str]:
    """
    Detect governed indicator codes from exact codes and common aliases.
    Longest aliases are checked first to avoid partial collisions.
    """
    text = query.lower()
    detected = []

    # Exact World Bank codes.
    for code in INDICATOR_REGISTRY:
        if code.lower() in text:
            detected.append(code)

    # Natural-language aliases.
    for alias, code in sorted(
        INDICATOR_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if alias in text and code not in detected:
            detected.append(code)

    return detected


assert detect_indicator_codes(
    "Show India's GDP growth since 2015"
) == ["NY.GDP.MKTP.KD.ZG"]

print("Deterministic indicator detection passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Deterministic year extraction

# COMMAND ----------

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def extract_years(query: str) -> List[int]:
    """Extract explicit four-digit years from the query."""
    return sorted({
        int(year)
        for year in YEAR_PATTERN.findall(query)
    })


assert extract_years(
    "Compare 2022 and 2026"
) == [2022, 2026]

print("Year extraction passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Fast deterministic route hints
# MAGIC
# MAGIC These are hints, not the final plan. The LLM resolves ambiguous language.

# COMMAND ----------

QUALITATIVE_TERMS = {
    "risk",
    "risks",
    "outlook",
    "forecast",
    "forecasts",
    "projected",
    "projection",
    "projections",
    "assessment",
    "policy",
    "policies",
    "challenge",
    "challenges",
    "headwind",
    "headwinds",
    "prospect",
    "prospects",
    "world bank identify",
    "world bank identified",
    "gep",
    "global economic prospects",
}

STRUCTURED_TERMS = {
    "show",
    "trend",
    "trends",
    "historical",
    "history",
    "actual",
    "value",
    "values",
    "from",
    "since",
    "between",
    "over time",
}


def deterministic_route_hint(
    query: str,
) -> Dict[str, Any]:
    """
    Produce a cheap routing hint.

    This does not execute any downstream tool.
    """
    text = query.lower()

    indicators = detect_indicator_codes(query)
    years = extract_years(query)

    has_structured_signal = bool(indicators)

    has_qualitative_signal = any(
        term in text
        for term in QUALITATIVE_TERMS
    )

    has_temporal_language = any(
        term in text
        for term in [
            "change between",
            "changed between",
            "compare the reports",
            "across reports",
            "between the 2022 and 2026",
            "between 2022 and 2026",
            "over the reports",
        ]
    )

    # Multiple explicit GEP-edition years plus qualitative language
    # is a strong temporal-RAG signal.
    gep_years = [
        year
        for year in years
        if year in VALID_REPORT_YEARS
    ]

    if (
        has_qualitative_signal
        and (
            has_temporal_language
            or len(gep_years) >= 2
        )
    ):
        route = "temporal_rag"

    elif (
        has_structured_signal
        and has_qualitative_signal
    ):
        route = "hybrid"

    elif has_structured_signal:
        route = "structured"

    elif has_qualitative_signal:
        route = "rag"

    else:
        route = "unknown"

    return {
        "route_hint": route,
        "detected_indicators": indicators,
        "explicit_years": years,
        "gep_year_candidates": gep_years,
    }


tests = [
    "Show India's GDP growth from 2015 to 2025",
    "What risks did the World Bank identify for South Asia in 2025?",
    (
        "Compare GDP growth in South Asia and Sub-Saharan Africa "
        "since 2022 and explain the World Bank outlook."
    ),
    (
        "How did the World Bank's assessment of global economic "
        "risks change between 2022 and 2026?"
    ),
]

for query in tests:
    print(query)
    print(
        json.dumps(
            deterministic_route_hint(query),
            indent=2,
        )
    )
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Supervisor prompt
# MAGIC
# MAGIC The Supervisor separates historical observation years from GEP publication years.

# COMMAND ----------

SUPERVISOR_SYSTEM_PROMPT = f"""
You are the routing supervisor for a World Bank economic intelligence application.

Your job is ONLY to create an execution plan.
Do not answer the user's economic question.
Do not invent data.
Do not invent report content.

Available downstream capabilities:

1. STRUCTURED DATA TOOLS
   Governed historical World Bank indicator observations.
   Historical observation years available: {MIN_OBSERVATION_YEAR}-{MAX_OBSERVATION_YEAR}.
   Supported indicators:
   {json.dumps(INDICATOR_REGISTRY, indent=2)}

2. GEP RESEARCH TOOL
   Retrieval over January Global Economic Prospects reports.
   Available report editions: 2022, 2023, 2024, 2025, 2026.
   Use for qualitative claims such as outlook, risks, policy discussion,
   report-specific projections, and comparisons of World Bank assessments.

Routes:
- structured:
  Historical indicator values/trends only.
- rag:
  Qualitative/report-grounded GEP research from one report edition or
  a general document question.
- hybrid:
  Requires BOTH historical indicator data and qualitative GEP evidence.
- temporal_rag:
  Compares qualitative GEP evidence across multiple report editions.
- unknown:
  Request is unsupported or too unclear to route safely.

CRITICAL YEAR RULE:
- observation_start_year / observation_end_year refer ONLY to historical
  indicator observations.
- report_years refer ONLY to GEP publication editions.
- Never put a historical observation range such as 2015-2025 into
  report_years merely because some years overlap available report editions.
- If the user asks "GDP growth from 2015 to 2025", that is structured
  observation data.
- If the user asks "risks in the 2025 GEP", 2025 is a report_year.
- If the user asks "how risks changed between 2022 and 2026",
  report_years are [2022, 2026].
- If a hybrid question says "GDP growth since 2022 and explain the outlook",
  2022 belongs to the structured observation range unless the wording also
  explicitly identifies a GEP edition.

FORECAST RULE:
Historical indicator tools stop at {MAX_OBSERVATION_YEAR}.
Do not route a request for future/projection values to structured historical
tools merely because the year is numeric. GEP forecast/projection questions
belong to RAG unless a separate forecast tool exists.

ENTITY RULE:
Extract explicitly named countries, economies, aggregates, or regions.
Do not invent an entity.

INDICATOR RULE:
Use only indicator codes from the supported registry.
If a requested indicator is not supported, leave indicators empty and explain
that limitation in reason.

Return ONLY valid JSON matching this structure:
{{
  "original_query": "...",
  "resolved_query": "...",
  "route": "structured|rag|hybrid|temporal_rag|unknown",
  "countries_or_entities": [],
  "indicators": [],
  "observation_start_year": null,
  "observation_end_year": null,
  "report_years": [],
  "regions": [],
  "topics": [],
  "is_temporal": false,
  "is_followup": false,
  "needs_structured_tool": false,
  "needs_research_tool": false,
  "reason": "short factual routing reason"
}}

Tool flags must agree with route:
structured -> structured=true, research=false
rag -> structured=false, research=true
hybrid -> structured=true, research=true
temporal_rag -> structured=false, research=true
unknown -> structured=false, research=false
""".strip()

print("Supervisor system prompt loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. JSON parsing helper

# COMMAND ----------

def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from the model response.
    Handles optional markdown code fences defensively.
    """
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Supervisor response must be a JSON object."
        )

    return parsed

print("JSON parser loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Plan validation and normalization
# MAGIC
# MAGIC We never trust the raw LLM route blindly. Deterministic checks enforce the contract.

# COMMAND ----------

def validate_supervisor_output(
    plan: SupervisorPlan,
) -> SupervisorPlan:
    """
    Enforce route/tool/year/indicator invariants.
    """
    data = model_to_dict(plan)

    route = data["route"]

    expected_flags = {
        "structured": (True, False),
        "rag": (False, True),
        "hybrid": (True, True),
        "temporal_rag": (False, True),
        "unknown": (False, False),
    }

    expected_structured, expected_research = (
        expected_flags[route]
    )

    # Deterministically correct tool flags.
    data["needs_structured_tool"] = expected_structured
    data["needs_research_tool"] = expected_research

    # Deduplicate while preserving order.
    data["countries_or_entities"] = list(
        dict.fromkeys(
            item.strip()
            for item in data["countries_or_entities"]
            if item and item.strip()
        )
    )

    data["regions"] = list(
        dict.fromkeys(
            item.strip()
            for item in data["regions"]
            if item and item.strip()
        )
    )

    # Structured tools resolve countries, economies, and World Bank aggregates
    # through countries_or_entities. If the Supervisor extracted an explicitly
    # named region only into `regions`, copy that same explicit value into the
    # structured entity list for structured/hybrid routes.
    #
    # This does NOT invent an entity; it only normalizes two fields in the
    # Supervisor contract.
    if route in {"structured", "hybrid"} and data["regions"]:
        data["countries_or_entities"] = list(
            dict.fromkeys(
                data["countries_or_entities"] + data["regions"]
            )
        )

    data["topics"] = list(
        dict.fromkeys(
            item.strip()
            for item in data["topics"]
            if item and item.strip()
        )
    )

    # Only governed indicator codes survive.
    data["indicators"] = list(
        dict.fromkeys(
            code
            for code in data["indicators"]
            if code in INDICATOR_REGISTRY
        )
    )

    # Only loaded GEP report editions survive.
    data["report_years"] = sorted({
        int(year)
        for year in data["report_years"]
        if int(year) in VALID_REPORT_YEARS
    })

    start_year = data["observation_start_year"]
    end_year = data["observation_end_year"]

    if start_year is not None:
        start_year = int(start_year)

    if end_year is not None:
        end_year = int(end_year)

    if (
        start_year is not None
        and start_year < MIN_OBSERVATION_YEAR
    ):
        raise ValueError(
            f"Observation start year {start_year} is before "
            f"the supported historical range {MIN_OBSERVATION_YEAR}."
        )

    if (
        end_year is not None
        and end_year > MAX_OBSERVATION_YEAR
        and route in {"structured", "hybrid"}
    ):
        # Do not silently pretend future historical observations exist.
        end_year = MAX_OBSERVATION_YEAR

    if (
        start_year is not None
        and end_year is not None
        and start_year > end_year
    ):
        raise ValueError(
            "observation_start_year cannot be greater than "
            "observation_end_year."
        )

    data["observation_start_year"] = start_year
    data["observation_end_year"] = end_year

    if route == "temporal_rag":
        data["is_temporal"] = True

        if len(data["report_years"]) < 2:
            raise ValueError(
                "temporal_rag requires at least two valid GEP report years."
            )

    if route == "structured":
        data["report_years"] = []

    return SupervisorPlan(**data)

print("Supervisor-plan validator loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Supervisor routing function

# COMMAND ----------

def supervisor_plan(
    query: str,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a user query into a validated downstream execution plan.

    No tools are executed here.
    No final answer is generated here.
    """
    if not query or not str(query).strip():
        raise ValueError("query cannot be empty.")

    query = str(query).strip()

    route_hint = deterministic_route_hint(query)

    context_text = (
        conversation_context.strip()
        if conversation_context
        else ""
    )

    user_payload = {
        "query": query,
        "conversation_context": context_text or None,
        "deterministic_hint": route_hint,
    }

    start = time.perf_counter()

    response = llm_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {
                "role": "system",
                "content": SUPERVISOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        temperature=0,
        max_tokens=900,
    )

    raw_text = response.choices[0].message.content

    parsed = parse_json_object(raw_text)

    # Ensure original query cannot be changed by the model.
    parsed["original_query"] = query

    plan = SupervisorPlan(**parsed)
    plan = validate_supervisor_output(plan)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    result = model_to_dict(plan)

    result["deterministic_hint"] = route_hint
    result["supervisor_latency_ms"] = round(
        latency_ms,
        2,
    )

    return result

print("supervisor_plan() loaded.")

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# Stable runtime alias used by evaluation and serving.
print("supervisor_runtime READY")
print(" - run_supervisor_agent")

# COMMAND ----------

# Stable runtime entry point.
# This wrapper guarantees a plain Python dictionary for downstream agents.
def run_supervisor_agent(
    query: str,
    conversation_context=None,
) -> Dict[str, Any]:
    result = supervisor_plan(
        query=query,
        conversation_context=conversation_context,
    )

    if isinstance(result, dict):
        return result

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "dict"):
        return result.dict()

    raise TypeError(
        "Supervisor returned unsupported type: "
        f"{type(result).__name__}"
    )


print("supervisor_runtime READY")
print(" - supervisor_plan")
print(" - run_supervisor_agent")

