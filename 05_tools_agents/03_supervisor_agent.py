# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 03_supervisor_agent
# MAGIC
# MAGIC Supervisor/router for the Global Economic Prospects Intelligence Agent.
# MAGIC
# MAGIC The Supervisor does **not** answer the user's question and does **not** execute tools.
# MAGIC It converts a natural-language request into a validated execution plan for downstream agents.
# MAGIC
# MAGIC ### Routes
# MAGIC - `structured` — historical macroeconomic observations from governed Gold tools
# MAGIC - `rag` — qualitative evidence from GEP reports
# MAGIC - `hybrid` — both structured historical data and GEP research evidence
# MAGIC - `temporal_rag` — comparison across multiple GEP report editions
# MAGIC - `unknown` — insufficiently clear or unsupported request
# MAGIC
# MAGIC ### Important separation
# MAGIC - `observation_years` are years for historical World Bank indicator data.
# MAGIC - `report_years` are GEP publication editions: 2022–2026.
# MAGIC
# MAGIC This fixes the ambiguity identified during Notebook 07 query-understanding work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Install model-serving SDK

# COMMAND ----------

# MAGIC %pip install -q databricks-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
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
# MAGIC ## 03. Validate model serving

# COMMAND ----------

health = llm_client.chat.completions.create(
    model=LLM_ENDPOINT,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly MODEL_OK",
        }
    ],
    temperature=0,
    max_tokens=10,
)

health_text = health.choices[0].message.content.strip()

assert "MODEL_OK" in health_text, (
    f"Model serving health check failed: {health_text}"
)

print("Model serving health check passed.")

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

def validate_supervisor_plan(
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
    plan = validate_supervisor_plan(plan)

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
# MAGIC ## 12. Structured routing test
# MAGIC
# MAGIC Expected:
# MAGIC - route = structured
# MAGIC - GDP growth indicator
# MAGIC - observation years = 2015–2025
# MAGIC - report_years = []

# COMMAND ----------

structured_test = supervisor_plan(
    "Show India's GDP growth from 2015 to 2025"
)

print(
    json.dumps(
        structured_test,
        indent=2,
    )
)

assert structured_test["route"] == "structured"
assert structured_test["needs_structured_tool"] is True
assert structured_test["needs_research_tool"] is False
assert "NY.GDP.MKTP.KD.ZG" in structured_test["indicators"]
assert structured_test["observation_start_year"] == 2015
assert structured_test["observation_end_year"] == 2025
assert structured_test["report_years"] == []

print("Structured routing test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. RAG routing test

# COMMAND ----------

rag_test = supervisor_plan(
    "What risks did the World Bank identify for South Asia in 2025?"
)

print(
    json.dumps(
        rag_test,
        indent=2,
    )
)

assert rag_test["route"] == "rag"
assert rag_test["needs_structured_tool"] is False
assert rag_test["needs_research_tool"] is True
assert rag_test["report_years"] == [2025]

print("RAG routing test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Hybrid routing test
# MAGIC
# MAGIC The year 2022 here belongs to the historical observation range.
# MAGIC It must not automatically become a GEP report-year filter.

# COMMAND ----------

hybrid_test = supervisor_plan(
    (
        "Compare GDP growth in South Asia and Sub-Saharan Africa "
        "since 2022 and explain the World Bank outlook for the two regions."
    )
)

print(
    json.dumps(
        hybrid_test,
        indent=2,
    )
)

assert hybrid_test["route"] == "hybrid"
assert hybrid_test["needs_structured_tool"] is True
assert hybrid_test["needs_research_tool"] is True
assert "NY.GDP.MKTP.KD.ZG" in hybrid_test["indicators"]
assert hybrid_test["observation_start_year"] == 2022
assert hybrid_test["observation_end_year"] in {
    None,
    MAX_OBSERVATION_YEAR,
}

print("Hybrid routing test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Temporal-RAG routing test

# COMMAND ----------

temporal_test = supervisor_plan(
    (
        "How did the World Bank's assessment of global economic risks "
        "change between 2022 and 2026?"
    )
)

print(
    json.dumps(
        temporal_test,
        indent=2,
    )
)

assert temporal_test["route"] == "temporal_rag"
assert temporal_test["needs_structured_tool"] is False
assert temporal_test["needs_research_tool"] is True
assert temporal_test["report_years"] == [2022, 2026]
assert temporal_test["is_temporal"] is True

print("Temporal-RAG routing test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Forecast guardrail test
# MAGIC
# MAGIC A request for a GEP projection should not be sent to historical structured tools.

# COMMAND ----------

forecast_test = supervisor_plan(
    "What does the 2026 GEP project for global economic growth?"
)

print(
    json.dumps(
        forecast_test,
        indent=2,
    )
)

assert forecast_test["route"] in {
    "rag",
    "temporal_rag",
}
assert forecast_test["needs_structured_tool"] is False
assert forecast_test["needs_research_tool"] is True
assert 2026 in forecast_test["report_years"]

print("Forecast routing guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Follow-up routing test
# MAGIC
# MAGIC Conversation context is supplied explicitly. The Supervisor may resolve the follow-up,
# MAGIC but it still does not execute any tool.

# COMMAND ----------

followup_test = supervisor_plan(
    query="What about 2026?",
    conversation_context=(
        "Previous user question: What risks did the World Bank identify "
        "for South Asia in the 2025 GEP? "
        "Previous route: rag. Region: South Asia. Report year: 2025."
    ),
)

print(
    json.dumps(
        followup_test,
        indent=2,
    )
)

assert followup_test["route"] in {
    "rag",
    "temporal_rag",
}
assert followup_test["needs_research_tool"] is True
assert 2026 in followup_test["report_years"]

print("Follow-up routing test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Agent-safe supervisor schema
# MAGIC
# MAGIC This is the interface the serving/orchestration layer can call later.

# COMMAND ----------

SUPERVISOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "supervisor_plan",
        "description": (
            "Create a validated execution plan for a World Bank economic "
            "intelligence query. Separates historical observation years "
            "from GEP publication report years and routes to structured, "
            "RAG, hybrid, temporal-RAG, or unknown. Does not execute tools "
            "and does not answer the user's question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's current question.",
                },
                "conversation_context": {
                    "type": "string",
                    "description": (
                        "Optional compact context from prior turns, used "
                        "only to resolve follow-up references."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

print(
    json.dumps(
        SUPERVISOR_TOOL_SCHEMA,
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Build downstream execution instructions
# MAGIC
# MAGIC This converts the validated route into an explicit orchestration plan.
# MAGIC It still does not execute Notebook 01 or Notebook 02 tools.

# COMMAND ----------

def build_execution_instructions(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Translate a validated Supervisor plan into downstream work items.
    """
    route = plan["route"]

    tasks = []

    if plan["needs_structured_tool"]:
        tasks.append({
            "agent": "data_agent",
            "purpose": "historical_structured_analysis",
            "countries_or_entities": plan[
                "countries_or_entities"
            ],
            "indicators": plan["indicators"],
            "observation_start_year": plan[
                "observation_start_year"
            ],
            "observation_end_year": plan[
                "observation_end_year"
            ],
        })

    if plan["needs_research_tool"]:
        tasks.append({
            "agent": "research_agent",
            "purpose": (
                "temporal_gep_research"
                if route == "temporal_rag"
                else "gep_research"
            ),
            "query": plan["resolved_query"],
            "report_years": plan["report_years"],
            "regions": plan["regions"],
            "topics": plan["topics"],
        })

    return {
        "route": route,
        "task_count": len(tasks),
        "tasks": tasks,
        "requires_synthesis": route in {
            "hybrid",
            "temporal_rag",
        },
    }


for name, plan in [
    ("structured", structured_test),
    ("rag", rag_test),
    ("hybrid", hybrid_test),
    ("temporal", temporal_test),
]:
    print("=" * 100)
    print(name.upper())
    print(
        json.dumps(
            build_execution_instructions(plan),
            indent=2,
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Final validation suite

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
    "structured_route",
    structured_test["route"] == "structured",
    structured_test["reason"],
)

record_test(
    "structured_year_separation",
    (
        structured_test["observation_start_year"] == 2015
        and structured_test["observation_end_year"] == 2025
        and structured_test["report_years"] == []
    ),
    "2015-2025 kept as observation years, not GEP editions",
)

record_test(
    "rag_route",
    (
        rag_test["route"] == "rag"
        and rag_test["report_years"] == [2025]
    ),
    rag_test["reason"],
)

record_test(
    "hybrid_route",
    (
        hybrid_test["route"] == "hybrid"
        and hybrid_test["needs_structured_tool"]
        and hybrid_test["needs_research_tool"]
    ),
    hybrid_test["reason"],
)

record_test(
    "temporal_rag_route",
    (
        temporal_test["route"] == "temporal_rag"
        and temporal_test["report_years"] == [2022, 2026]
    ),
    temporal_test["reason"],
)

record_test(
    "forecast_guardrail",
    (
        forecast_test["needs_structured_tool"] is False
        and forecast_test["needs_research_tool"] is True
    ),
    forecast_test["reason"],
)

record_test(
    "followup_resolution",
    (
        followup_test["needs_research_tool"] is True
        and 2026 in followup_test["report_years"]
    ),
    followup_test["resolved_query"],
)

# Check route/tool flag consistency across all tested plans.
tested_plans = [
    structured_test,
    rag_test,
    hybrid_test,
    temporal_test,
    forecast_test,
    followup_test,
]

expected_flags = {
    "structured": (True, False),
    "rag": (False, True),
    "hybrid": (True, True),
    "temporal_rag": (False, True),
    "unknown": (False, False),
}

flags_consistent = all(
    (
        plan["needs_structured_tool"],
        plan["needs_research_tool"],
    )
    == expected_flags[plan["route"]]
    for plan in tested_plans
)

record_test(
    "route_tool_consistency",
    flags_consistent,
    "All tested routes map to deterministic downstream tool flags",
)

validation_df = spark.createDataFrame(
    validation_results
)

display(validation_df)

failed_tests = [
    row["test"]
    for row in validation_results
    if not row["passed"]
]

assert not failed_tests, (
    f"Supervisor validation failures: {failed_tests}"
)

print("All Supervisor validation tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Final status

# COMMAND ----------

print("")
print("03_supervisor_agent COMPLETE")
print("")
print("Validated responsibility:")
print(
    "user query -> deterministic hints -> LLM planning -> "
    "schema validation -> route/tool invariants -> execution instructions"
)
print("")
print("Supported routes:")
print(" - structured")
print(" - rag")
print(" - hybrid")
print(" - temporal_rag")
print(" - unknown")
print("")
print("Important:")
print(" - Supervisor does not answer the question")
print(" - Supervisor does not execute tools")
print(" - Observation years and GEP report years are separate")
print(" - Historical structured data stops at 2025")
print(" - GEP projection questions route to research")
print(" - Downstream tool flags are enforced deterministically")
print("")
print("Next notebook after this passes:")
print("05_tools_agents / 04_data_agent")