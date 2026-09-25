# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 07_query_understanding
# MAGIC
# MAGIC Purpose:
# MAGIC - Convert a user's natural-language question into validated retrieval intent.
# MAGIC - Extract reliable report-year and region filters deterministically.
# MAGIC - Use an LLM only when the request is ambiguous, temporal, or a follow-up.
# MAGIC - Preserve conversational context without sending the entire chat history.
# MAGIC - Produce a typed object that Notebook 08 can pass to AI Search.
# MAGIC
# MAGIC This notebook does NOT perform retrieval or answer the user's question.

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-openai pydantic mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Imports and configuration

# COMMAND ----------

import json
import re
import time
from enum import Enum
from typing import List, Optional, Dict, Any

import mlflow
from pydantic import BaseModel, Field, field_validator
from databricks_openai import DatabricksOpenAI

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

SUPPORTED_REPORT_YEARS = [2022, 2023, 2024, 2025, 2026]

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_query_understanding"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

llm_client = DatabricksOpenAI()

print("Query-understanding configuration loaded.")
print("LLM endpoint:", LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Canonical enums and output schema
# MAGIC
# MAGIC `route_hint` is intentionally only a hint.
# MAGIC The future Supervisor Agent will make the final structured/RAG/hybrid/temporal routing decision.

# COMMAND ----------

class RouteHint(str, Enum):
    STRUCTURED = "structured"
    RAG = "rag"
    HYBRID = "hybrid"
    TEMPORAL_RAG = "temporal_rag"
    UNKNOWN = "unknown"


class QueryUnderstanding(BaseModel):
    original_query: str

    # Standalone query after resolving a conversational follow-up.
    resolved_query: str

    route_hint: RouteHint = RouteHint.UNKNOWN

    # GEP report-year filters.
    report_years: List[int] = Field(default_factory=list)

    # Canonical region names used by our corpus.
    regions: List[str] = Field(default_factory=list)

    # Optional document-structure hints.
    sections: List[str] = Field(default_factory=list)

    # Country/economy mentions are retained for later routing/tool use.
    countries: List[str] = Field(default_factory=list)

    # Macro indicator concepts retained for the structured-data route.
    indicators: List[str] = Field(default_factory=list)

    # Whether the query explicitly compares multiple report editions/time periods.
    is_temporal_comparison: bool = False

    # Whether current user text depends on prior conversation.
    is_followup: bool = False

    # Whether deterministic parsing was sufficient.
    used_llm: bool = False

    # Short machine-readable reason for debugging/tracing.
    interpretation_reason: str = ""

    @field_validator("report_years")
    @classmethod
    def validate_years(cls, years):
        clean = sorted(set(int(y) for y in years))

        invalid = [
            y for y in clean
            if y not in SUPPORTED_REPORT_YEARS
        ]

        if invalid:
            raise ValueError(
                f"Unsupported GEP report year(s): {invalid}"
            )

        return clean

    @field_validator("regions")
    @classmethod
    def deduplicate_regions(cls, regions):
        return list(dict.fromkeys(regions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Canonical region aliases
# MAGIC
# MAGIC These map common user wording to the region labels already used in the GEP corpus.

# COMMAND ----------

REGION_ALIASES = {
    "South Asia": [
        "south asia",
        "south asian",
    ],
    "Sub-Saharan Africa": [
        "sub-saharan africa",
        "sub saharan africa",
        "ssa",
    ],
    "East Asia and Pacific": [
        "east asia and pacific",
        "east asia & pacific",
        "eap",
    ],
    "Europe and Central Asia": [
        "europe and central asia",
        "europe & central asia",
        "eca",
    ],
    "Latin America and the Caribbean": [
        "latin america and the caribbean",
        "latin america & caribbean",
        "latin america",
        "lac",
    ],
    "Middle East and North Africa": [
        "middle east and north africa",
        "middle east & north africa",
        "mena",
    ],
}

SECTION_ALIASES = {
    "Risks": [
        "risk",
        "risks",
        "downside risk",
        "downside risks",
        "risk to the outlook",
        "risks to the outlook",
    ],
    "Outlook": [
        "outlook",
        "forecast",
        "prospects",
    ],
    "Recent developments": [
        "recent developments",
        "recent development",
    ],
    "Policy priorities": [
        "policy priorities",
        "policy priority",
    ],
    "Policy implications": [
        "policy implications",
        "policy implication",
    ],
}

# Concepts that strongly suggest the governed structured-data route.
INDICATOR_ALIASES = {
    "GDP growth": [
        "gdp growth",
        "economic growth",
    ],
    "GDP per capita growth": [
        "gdp per capita growth",
    ],
    "GDP": [
        "gdp",
        "gross domestic product",
    ],
    "GDP per capita": [
        "gdp per capita",
    ],
    "Inflation": [
        "inflation",
        "consumer prices",
        "cpi",
    ],
    "Trade": [
        "trade",
    ],
    "Exports": [
        "exports",
    ],
    "Imports": [
        "imports",
    ],
    "Gross capital formation": [
        "gross capital formation",
        "investment share",
    ],
    "Government debt": [
        "government debt",
        "central government debt",
        "debt to gdp",
    ],
    "Domestic savings": [
        "domestic savings",
    ],
    "FDI": [
        "foreign direct investment",
        "fdi",
    ],
    "Current account balance": [
        "current account",
        "current account balance",
    ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Deterministic extraction helpers
# MAGIC
# MAGIC We do not call an LLM for information that can be extracted safely with rules.

# COMMAND ----------

def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for deterministic matching."""

    text = text or ""
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_report_years(text: str) -> List[int]:
    """Extract explicit GEP edition years supported by our corpus."""

    years = {
        int(year)
        for year in re.findall(r"\b20\d{2}\b", text)
        if int(year) in SUPPORTED_REPORT_YEARS
    }

    return sorted(years)


def extract_regions(text: str) -> List[str]:
    """Map region aliases to canonical corpus region labels."""

    lower = text.lower()
    found = []

    for canonical, aliases in REGION_ALIASES.items():
        for alias in aliases:
            if re.search(
                rf"\b{re.escape(alias)}\b",
                lower,
                flags=re.IGNORECASE,
            ):
                found.append(canonical)
                break

    return list(dict.fromkeys(found))


def extract_sections(text: str) -> List[str]:
    """Extract broad GEP section intent."""

    lower = text.lower()
    found = []

    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if re.search(
                rf"\b{re.escape(alias)}\b",
                lower,
                flags=re.IGNORECASE,
            ):
                found.append(canonical)
                break

    return list(dict.fromkeys(found))


def extract_indicators(text: str) -> List[str]:
    """Extract structured macroeconomic concepts."""

    lower = text.lower()

    # Match longer/specific concepts first.
    candidates = sorted(
        INDICATOR_ALIASES.items(),
        key=lambda item: max(len(x) for x in item[1]),
        reverse=True,
    )

    found = []

    for canonical, aliases in candidates:
        for alias in aliases:
            if re.search(
                rf"\b{re.escape(alias)}\b",
                lower,
                flags=re.IGNORECASE,
            ):
                found.append(canonical)
                break

    # Avoid returning both GDP and GDP growth when the specific concept matched.
    if "GDP growth" in found and "GDP" in found:
        found.remove("GDP")

    if "GDP per capita growth" in found:
        found = [
            x for x in found
            if x not in {"GDP per capita", "GDP growth", "GDP"}
        ] + ["GDP per capita growth"]

    return list(dict.fromkeys(found))


def detect_temporal_comparison(
    text: str,
    report_years: List[int],
) -> bool:
    """Detect explicit comparison across report vintages."""

    lower = text.lower()

    comparison_terms = [
        "change between",
        "changed between",
        "compare",
        "comparison",
        "evolve",
        "evolved",
        "over time",
        "from 202",
        "between 202",
    ]

    return (
        len(report_years) >= 2
        and any(term in lower for term in comparison_terms)
    )


def looks_like_followup(text: str) -> bool:
    """
    Detect short/context-dependent follow-ups.

    Examples:
    - What about 2026?
    - And South Asia?
    - How about the risks?
    """

    lower = text.lower().strip()

    followup_starters = (
        "what about",
        "how about",
        "and ",
        "what if",
        "then ",
        "also ",
    )

    pronouns = [
        "that",
        "those",
        "them",
        "same",
        "it",
    ]

    return (
        lower.startswith(followup_starters)
        or (
            len(lower.split()) <= 8
            and any(
                re.search(rf"\b{word}\b", lower)
                for word in pronouns
            )
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Deterministic route hint
# MAGIC
# MAGIC This is not the final Supervisor router.
# MAGIC It only gives downstream components an inexpensive first signal.

# COMMAND ----------

def deterministic_route_hint(
    query: str,
    report_years: List[int],
    sections: List[str],
    indicators: List[str],
    temporal: bool,
) -> RouteHint:

    lower = query.lower()

    qualitative_terms = [
        "world bank",
        "gep",
        "global economic prospects",
        "risk",
        "risks",
        "outlook",
        "prospects",
        "assessment",
        "policy",
        "report",
        "identified",
        "said",
        "discussed",
    ]

    structured_terms = [
        "show",
        "trend",
        "historical",
        "from 20",
        "since 20",
        "percentage",
        "value",
    ]

    has_qualitative = (
        bool(sections)
        or any(term in lower for term in qualitative_terms)
    )

    has_structured = (
        bool(indicators)
        and any(term in lower for term in structured_terms)
    )

    if temporal and has_qualitative:
        return RouteHint.TEMPORAL_RAG

    if has_structured and has_qualitative:
        return RouteHint.HYBRID

    if has_structured:
        return RouteHint.STRUCTURED

    if has_qualitative or report_years:
        return RouteHint.RAG

    return RouteHint.UNKNOWN

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Decide whether the LLM is necessary
# MAGIC
# MAGIC LLM interpretation is reserved for cases such as:
# MAGIC - conversational follow-ups;
# MAGIC - ambiguous intent;
# MAGIC - complex mixed structured + qualitative questions;
# MAGIC - temporal comparisons requiring interpretation.

# COMMAND ----------

def should_use_llm(
    query: str,
    route_hint: RouteHint,
    is_followup: bool,
    is_temporal: bool,
) -> bool:

    if is_followup:
        return True

    if route_hint in {
        RouteHint.UNKNOWN,
        RouteHint.HYBRID,
        RouteHint.TEMPORAL_RAG,
    }:
        return True

    # Long multi-clause questions can benefit from semantic interpretation.
    if len(query.split()) >= 25:
        return True

    return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. LLM structured interpretation
# MAGIC
# MAGIC The LLM is allowed to interpret intent, but not invent unsupported report years or metadata.

# COMMAND ----------

QUERY_SYSTEM_PROMPT = """
You are the query-understanding component for a World Bank economic intelligence application.

The application has:
1. Structured historical World Bank macroeconomic indicators.
2. Global Economic Prospects January reports for 2022, 2023, 2024, 2025, and 2026.

Your task is ONLY to interpret the user's request.
Do not answer the economic question.

Allowed route_hint values:
- structured
- rag
- hybrid
- temporal_rag
- unknown

Use:
- structured: historical numerical indicator lookup/trend
- rag: qualitative evidence from one GEP report or report content
- hybrid: needs both historical numerical data and GEP qualitative evidence
- temporal_rag: compares qualitative GEP assessments across multiple report editions
- unknown: insufficient information

Canonical GEP regions:
- South Asia
- Sub-Saharan Africa
- East Asia and Pacific
- Europe and Central Asia
- Latin America and the Caribbean
- Middle East and North Africa

Rules:
- Never invent a report year.
- Only GEP report years 2022-2026 are supported.
- Preserve the user's meaning.
- Resolve a follow-up using conversation context when supplied.
- Do not add economic facts.
- Do not answer the question.
- Return valid JSON only.

JSON format:
{
  "resolved_query": "...",
  "route_hint": "structured|rag|hybrid|temporal_rag|unknown",
  "report_years": [],
  "regions": [],
  "sections": [],
  "countries": [],
  "indicators": [],
  "is_temporal_comparison": false,
  "interpretation_reason": "short reason"
}
""".strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    """Strict but fence-tolerant JSON parser."""

    if not text:
        raise ValueError("LLM returned empty output.")

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Could not locate JSON object in LLM output: {cleaned}"
        )

    return json.loads(cleaned[start:end + 1])


def llm_interpret_query(
    query: str,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:

    context = conversation_context or []

    # Keep only recent conversational context.
    # We do not send unlimited chat history.
    recent_context = context[-4:]

    payload = {
        "current_user_query": query,
        "recent_conversation": recent_context,
    }

    last_error = None

    for attempt in range(max_retries + 1):

        try:
            response = llm_client.chat.completions.create(
                model=LLM_ENDPOINT,
                messages=[
                    {
                        "role": "system",
                        "content": QUERY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0,
                max_tokens=700,
            )

            raw = response.choices[0].message.content

            return parse_json_object(raw)

        except Exception as exc:
            last_error = exc

            print(
                f"Query-understanding LLM attempt "
                f"{attempt + 1}/{max_retries + 1} failed: {exc}"
            )

            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(
        "LLM query understanding failed after retries."
    ) from last_error

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Main query-understanding function
# MAGIC
# MAGIC Deterministic extraction has priority for explicit years and regions.
# MAGIC The LLM supplements ambiguity rather than silently overriding reliable fields.

# COMMAND ----------

def understand_query(
    query: str,
    conversation_context: Optional[List[Dict[str, str]]] = None,
) -> QueryUnderstanding:

    original_query = normalize_text(query)

    if not original_query:
        raise ValueError("Query cannot be empty.")

    # ----------------------------------
    # Deterministic extraction
    # ----------------------------------
    deterministic_years = extract_report_years(original_query)
    deterministic_regions = extract_regions(original_query)
    deterministic_sections = extract_sections(original_query)
    deterministic_indicators = extract_indicators(original_query)

    temporal = detect_temporal_comparison(
        original_query,
        deterministic_years,
    )

    followup = looks_like_followup(original_query)

    route = deterministic_route_hint(
        query=original_query,
        report_years=deterministic_years,
        sections=deterministic_sections,
        indicators=deterministic_indicators,
        temporal=temporal,
    )

    use_llm = should_use_llm(
        query=original_query,
        route_hint=route,
        is_followup=followup,
        is_temporal=temporal,
    )

    # ----------------------------------
    # Deterministic-only path
    # ----------------------------------
    if not use_llm:

        return QueryUnderstanding(
            original_query=original_query,
            resolved_query=original_query,
            route_hint=route,
            report_years=deterministic_years,
            regions=deterministic_regions,
            sections=deterministic_sections,
            countries=[],
            indicators=deterministic_indicators,
            is_temporal_comparison=temporal,
            is_followup=followup,
            used_llm=False,
            interpretation_reason="deterministic_extraction_sufficient",
        )

    # ----------------------------------
    # LLM-assisted path
    # ----------------------------------
    llm_result = llm_interpret_query(
        query=original_query,
        conversation_context=conversation_context,
    )

    llm_years = [
        int(y)
        for y in llm_result.get("report_years", [])
        if int(y) in SUPPORTED_REPORT_YEARS
    ]

    llm_regions = [
        str(x)
        for x in llm_result.get("regions", [])
        if str(x) in REGION_ALIASES
    ]

    # Explicit values in the current query take priority.
    final_years = (
        deterministic_years
        if deterministic_years
        else llm_years
    )

    final_regions = (
        deterministic_regions
        if deterministic_regions
        else llm_regions
    )

    final_sections = (
        deterministic_sections
        if deterministic_sections
        else [
            str(x)
            for x in llm_result.get("sections", [])
        ]
    )

    final_indicators = (
        deterministic_indicators
        if deterministic_indicators
        else [
            str(x)
            for x in llm_result.get("indicators", [])
        ]
    )

    route_value = llm_result.get(
        "route_hint",
        route.value,
    )

    try:
        final_route = RouteHint(route_value)
    except ValueError:
        final_route = route

    return QueryUnderstanding(
        original_query=original_query,
        resolved_query=normalize_text(
            llm_result.get(
                "resolved_query",
                original_query,
            )
        ),
        route_hint=final_route,
        report_years=final_years,
        regions=final_regions,
        sections=final_sections,
        countries=[
            str(x)
            for x in llm_result.get("countries", [])
        ],
        indicators=final_indicators,
        is_temporal_comparison=bool(
            llm_result.get(
                "is_temporal_comparison",
                temporal,
            )
        ),
        is_followup=followup,
        used_llm=True,
        interpretation_reason=str(
            llm_result.get(
                "interpretation_reason",
                "llm_assisted_interpretation",
            )
        ),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Deterministic unit-style tests

# COMMAND ----------

# Explicit report year + region + risk intent.
q1 = understand_query(
    "What risks did the World Bank identify for South Asia in 2025?"
)

print(q1.model_dump_json(indent=2))

assert q1.report_years == [2025]
assert "South Asia" in q1.regions
assert "Risks" in q1.sections

# Straightforward historical numerical request.
q2 = understand_query(
    "Show GDP growth from 2022 to 2025."
)

print(q2.model_dump_json(indent=2))

assert "GDP growth" in q2.indicators

print("Basic query-understanding tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Conversational follow-up test
# MAGIC
# MAGIC The current question alone is incomplete, so this should invoke the LLM.

# COMMAND ----------

conversation = [
    {
        "role": "user",
        "content": (
            "What risks did the World Bank identify "
            "for South Asia in 2025?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The prior answer discussed the 2025 South Asia "
            "GEP risk assessment."
        ),
    },
]

followup_result = understand_query(
    "What about 2026?",
    conversation_context=conversation,
)

print(
    followup_result.model_dump_json(indent=2)
)

assert followup_result.is_followup is True
assert followup_result.used_llm is True
assert followup_result.report_years == [2026]
assert "South Asia" in followup_result.regions

print("Follow-up resolution passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Temporal comparison test

# COMMAND ----------

temporal_result = understand_query(
    (
        "How did the World Bank's assessment of global "
        "economic risks change between 2022 and 2026?"
    )
)

print(
    temporal_result.model_dump_json(indent=2)
)

assert temporal_result.report_years == [2022, 2026]
assert temporal_result.is_temporal_comparison is True
assert temporal_result.route_hint == RouteHint.TEMPORAL_RAG

print("Temporal query interpretation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Hybrid query test

# COMMAND ----------

hybrid_result = understand_query(
    (
        "Compare GDP growth in South Asia and "
        "Sub-Saharan Africa since 2022 and explain "
        "the World Bank outlook for the two regions."
    )
)

print(
    hybrid_result.model_dump_json(indent=2)
)

assert "GDP growth" in hybrid_result.indicators
assert "South Asia" in hybrid_result.regions
assert "Sub-Saharan Africa" in hybrid_result.regions
assert hybrid_result.route_hint == RouteHint.HYBRID

print("Hybrid interpretation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Small evaluation suite
# MAGIC
# MAGIC This checks extraction/routing behavior before connecting the component to online retrieval.

# COMMAND ----------

TEST_CASES = [
    {
        "query": "What risks did the World Bank identify for South Asia in 2025?",
        "expected_years": [2025],
        "expected_regions": ["South Asia"],
        "expected_route": "rag",
    },
    {
        "query": "What was the outlook for Sub-Saharan Africa in the 2024 GEP?",
        "expected_years": [2024],
        "expected_regions": ["Sub-Saharan Africa"],
        "expected_route": "rag",
    },
    {
        "query": "How did global economic risks change between 2022 and 2026?",
        "expected_years": [2022, 2026],
        "expected_regions": [],
        "expected_route": "temporal_rag",
    },
    {
        "query": "Show GDP growth from 2022 to 2025.",
        "expected_years": [2022, 2025],
        "expected_regions": [],
        "expected_route": "structured",
    },
    {
        "query": (
            "Compare GDP growth in South Asia and Sub-Saharan Africa "
            "since 2022 and explain the World Bank outlook."
        ),
        "expected_years": [2022],
        "expected_regions": [
            "South Asia",
            "Sub-Saharan Africa",
        ],
        "expected_route": "hybrid",
    },
]

evaluation_rows = []

for case in TEST_CASES:

    result = understand_query(case["query"])

    year_ok = (
        result.report_years
        == case["expected_years"]
    )

    region_ok = (
        set(result.regions)
        == set(case["expected_regions"])
    )

    route_ok = (
        result.route_hint.value
        == case["expected_route"]
    )

    evaluation_rows.append({
        "query": case["query"],
        "actual_years": result.report_years,
        "expected_years": case["expected_years"],
        "year_ok": year_ok,
        "actual_regions": result.regions,
        "expected_regions": case["expected_regions"],
        "region_ok": region_ok,
        "actual_route": result.route_hint.value,
        "expected_route": case["expected_route"],
        "route_ok": route_ok,
        "used_llm": result.used_llm,
    })

evaluation_df = spark.createDataFrame(evaluation_rows)

display(evaluation_df)

# We deliberately fail the notebook if the contract is broken.
failed = evaluation_df.filter(
    (~evaluation_df.year_ok)
    | (~evaluation_df.region_ok)
    | (~evaluation_df.route_ok)
).count()

assert failed == 0, (
    f"{failed} query-understanding test case(s) failed."
)

print("Query-understanding evaluation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. MLflow evaluation logging

# COMMAND ----------

total_tests = len(evaluation_rows)

year_accuracy = (
    sum(int(row["year_ok"]) for row in evaluation_rows)
    / total_tests
)

region_accuracy = (
    sum(int(row["region_ok"]) for row in evaluation_rows)
    / total_tests
)

route_accuracy = (
    sum(int(row["route_ok"]) for row in evaluation_rows)
    / total_tests
)

llm_usage_rate = (
    sum(int(row["used_llm"]) for row in evaluation_rows)
    / total_tests
)

with mlflow.start_run(
    run_name="query_understanding_v1"
):

    mlflow.log_params({
        "llm_endpoint": LLM_ENDPOINT,
        "supported_gep_years": ",".join(
            map(str, SUPPORTED_REPORT_YEARS)
        ),
        "strategy": "deterministic_plus_conditional_llm",
        "conversation_window_turns": 4,
    })

    mlflow.log_metrics({
        "test_count": total_tests,
        "year_accuracy": year_accuracy,
        "region_accuracy": region_accuracy,
        "route_accuracy": route_accuracy,
        "llm_usage_rate": llm_usage_rate,
    })

print("MLflow evaluation logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Final output contract demonstration
# MAGIC
# MAGIC Notebook 08 will consume this object and translate validated fields into AI Search filters.

# COMMAND ----------

demo = understand_query(
    "What risks did the World Bank identify for South Asia in 2025?"
)

print("ONLINE RETRIEVAL INPUT")
print("----------------------")
print(demo.model_dump_json(indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Completion

# COMMAND ----------

print("07_query_understanding COMPLETE")
print("")
print("Implemented:")
print("- deterministic year extraction")
print("- deterministic GEP region normalization")
print("- section/indicator extraction")
print("- lightweight route hinting")
print("- conditional LLM interpretation")
print("- conversational follow-up resolution")
print("- temporal comparison detection")
print("- validated Pydantic output contract")
print("- evaluation + MLflow logging")
print("")
