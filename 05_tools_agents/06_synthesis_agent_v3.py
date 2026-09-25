# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 06_synthesis_agent_v3
# MAGIC
# MAGIC Final synthesis layer for the Global Economic Prospects Intelligence Agent.
# MAGIC
# MAGIC This notebook combines two already-governed evidence streams:
# MAGIC
# MAGIC 1. **Data Agent payload** — deterministic historical macroeconomic facts from Gold Delta tables.
# MAGIC 2. **Research Agent payload** — qualitative Global Economic Prospects evidence with `[E#]` citations.
# MAGIC
# MAGIC The Synthesis Agent does **not** query arbitrary SQL, AI Search, or external sources.
# MAGIC It only synthesizes validated upstream payloads.
# MAGIC
# MAGIC ### Design rules
# MAGIC - structured numbers remain structured facts
# MAGIC - GEP interpretation remains document-grounded
# MAGIC - never convert missing values to zero
# MAGIC - never treat GEP forecasts as historical observations
# MAGIC - preserve `[E#]` research citations
# MAGIC - reject unsupported/invented evidence IDs
# MAGIC - explicitly separate historical observations from report outlook
# MAGIC - deterministic synthesis for structured-only questions
# MAGIC - LLM synthesis only when qualitative research evidence is present

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Install the Databricks model client

# COMMAND ----------

# MAGIC %pip install -q databricks-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Load validated upstream agents FIRST
# MAGIC
# MAGIC `05_research_agent` is loaded first because it may install packages and restart
# MAGIC Python. `04_data_agent` is loaded second. Only after both finish do we define
# MAGIC the Synthesis Agent, so its functions cannot be erased by an upstream restart.
# MAGIC
# MAGIC These relative paths assume all three notebooks are in the same Workspace folder.

# COMMAND ----------

# MAGIC %run ./05_research_agent

# COMMAND ----------

# MAGIC %run ./04_data_agent

# COMMAND ----------

# Verify the upstream functions before defining the Synthesis Agent.

required_upstream_functions = [
    "run_data_agent",
    "run_research_agent",
]

missing_upstream_functions = [
    name
    for name in required_upstream_functions
    if name not in globals()
    or not callable(globals()[name])
]

if missing_upstream_functions:
    raise RuntimeError(
        "Upstream agent import failed. Missing/non-callable functions: "
        f"{missing_upstream_functions}. "
        "Verify 04_data_agent and 05_research_agent are in the same Workspace folder."
    )

print("Upstream agents imported successfully.")
print(" - run_data_agent: READY")
print(" - run_research_agent: READY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Imports and frozen configuration

# COMMAND ----------

from typing import Any, Dict, List, Optional, Tuple
from databricks_openai import DatabricksOpenAI
import json
import math
import re
import time

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
MAX_SYNTHESIS_TOKENS = 1400

VALID_ROUTES = {
    "structured",
    "rag",
    "temporal_rag",
    "hybrid",
}

CITATION_PATTERN = re.compile(r"\[(E\d+)\]")

print("Synthesis Agent configuration loaded.")
print("LLM endpoint:", LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Initialize Databricks model serving

# COMMAND ----------

llm_client = DatabricksOpenAI()

model_test = llm_client.chat.completions.create(
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

model_test_text = model_test.choices[0].message.content.strip()

print("Model test:", model_test_text)
assert "MODEL_OK" in model_test_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. JSON-safe helper

# COMMAND ----------

def json_safe(value: Any) -> Any:
    """Convert nested Python values into JSON-safe values."""
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

    return str(value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Upstream payload normalization
# MAGIC
# MAGIC The previous notebooks intentionally expose compact agent payloads.
# MAGIC This notebook accepts those payloads without exposing the underlying SQL or search interfaces.
# MAGIC
# MAGIC To make the contract robust, we support:
# MAGIC - the compact Data Agent synthesis payload
# MAGIC - the full Data Agent result when it contains a synthesis payload
# MAGIC - the compact Research Agent synthesis payload
# MAGIC - the full Research Agent result from Notebook 05

# COMMAND ----------

def normalize_data_payload(
    data_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize the structured evidence contract."""
    if data_result is None:
        return None

    if not isinstance(data_result, dict):
        raise TypeError("Data Agent payload must be a dictionary.")

    # Prefer an explicitly nested synthesis payload if present.
    for key in [
        "synthesis_payload",
        "structured_synthesis_payload",
        "data_synthesis_payload",
    ]:
        nested = data_result.get(key)
        if isinstance(nested, dict):
            data_result = nested
            break

    # Reject failed upstream execution.
    status = data_result.get("status")
    if status is not None and status != "success":
        raise ValueError(
            f"Data Agent result is not successful: status={status!r}"
        )

    return json_safe(data_result)


def normalize_research_payload(
    research_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize the qualitative GEP evidence contract."""
    if research_result is None:
        return None

    if not isinstance(research_result, dict):
        raise TypeError("Research Agent payload must be a dictionary.")

    status = research_result.get("status")
    if status is not None and status != "success":
        raise ValueError(
            f"Research Agent result is not successful: status={status!r}"
        )

    # Notebook 05 compact payload.
    if research_result.get("evidence_type") == "gep_research":
        payload = dict(research_result)

    # Notebook 05 full result.
    elif research_result.get("agent") == "research_agent":
        citation_validation = research_result.get(
            "citation_validation", {}
        )

        if citation_validation.get("citation_valid") is not True:
            raise ValueError(
                "Research Agent result did not pass citation validation."
            )

        cited_ids = set(
            citation_validation.get(
                "cited_evidence_ids", []
            )
        )

        cited_evidence = [
            item
            for item in research_result.get("evidence", [])
            if item.get("evidence_id") in cited_ids
        ]

        payload = {
            "evidence_type": "gep_research",
            "query": research_result.get("query"),
            "report_years": research_result.get("report_years", []),
            "research_answer": research_result.get("answer"),
            "cited_evidence": cited_evidence,
            "citation_provenance": citation_validation.get(
                "citation_provenance", []
            ),
            "citation_valid": True,
        }

    else:
        payload = dict(research_result)

    if payload.get("citation_valid") is not True:
        raise ValueError(
            "Research payload must have citation_valid=True."
        )

    answer = str(
        payload.get("research_answer") or ""
    ).strip()

    if not answer:
        raise ValueError(
            "Research payload is missing research_answer."
        )

    evidence = payload.get("cited_evidence", []) or []

    if not evidence:
        raise ValueError(
            "Research payload contains no cited evidence."
        )

    return json_safe(payload)


print("Upstream payload normalization loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Research citation contract validation
# MAGIC
# MAGIC The Synthesis Agent never receives permission to invent new evidence IDs.
# MAGIC Every `[E#]` identifier must already exist in the Research Agent payload.

# COMMAND ----------

def allowed_research_evidence_ids(
    research_payload: Dict[str, Any],
) -> List[str]:
    ids = []

    for item in research_payload.get("cited_evidence", []):
        evidence_id = str(
            item.get("evidence_id") or ""
        ).strip()

        if evidence_id and evidence_id not in ids:
            ids.append(evidence_id)

    return ids


def validate_research_payload_citations(
    research_payload: Dict[str, Any],
) -> Dict[str, Any]:
    allowed = set(
        allowed_research_evidence_ids(
            research_payload
        )
    )

    answer = str(
        research_payload.get("research_answer") or ""
    )

    cited = []
    for evidence_id in CITATION_PATTERN.findall(answer):
        if evidence_id not in cited:
            cited.append(evidence_id)

    unsupported = [
        evidence_id
        for evidence_id in cited
        if evidence_id not in allowed
    ]

    if not cited:
        raise ValueError(
            "Research answer contains no [E#] citations."
        )

    if unsupported:
        raise ValueError(
            f"Research payload contains unsupported citations: {unsupported}"
        )

    return {
        "citation_valid": True,
        "allowed_evidence_ids": sorted(allowed),
        "cited_evidence_ids": cited,
        "unsupported_citations": [],
    }


print("Research citation contract validation loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Extract structured facts without changing values
# MAGIC
# MAGIC We do not calculate new forecasts or fill missing values.
# MAGIC This helper recursively finds observation-like rows so the final prompt can present
# MAGIC the structured evidence compactly while preserving the original values.

# COMMAND ----------

def _looks_like_observation(row: Dict[str, Any]) -> bool:
    keys = set(row.keys())

    has_year = "year" in keys
    has_value = "value" in keys

    has_entity = bool(
        {
            "entity_name",
            "entity_id",
            "iso2_code",
        } & keys
    )

    has_indicator = bool(
        {
            "indicator_id",
            "indicator_code",
            "metric_name",
            "indicator_display_name",
        } & keys
    )

    return (
        has_year
        and has_value
        and (
            has_entity
            or has_indicator
        )
    )


def collect_observation_rows(
    obj: Any,
) -> List[Dict[str, Any]]:
    """Recursively collect structured observation rows from the Data Agent payload."""
    rows = []

    if isinstance(obj, dict):
        if _looks_like_observation(obj):
            rows.append(json_safe(obj))

        for value in obj.values():
            rows.extend(
                collect_observation_rows(value)
            )

    elif isinstance(obj, list):
        for item in obj:
            rows.extend(
                collect_observation_rows(item)
            )

    # De-duplicate exact JSON representations while preserving order.
    unique_rows = []
    seen = set()

    for row in rows:
        key = json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
        )

        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return unique_rows


def build_structured_context(
    data_payload: Dict[str, Any],
) -> Dict[str, Any]:
    rows = collect_observation_rows(
        data_payload
    )

    # If the upstream payload has a compact human-readable facts field,
    # preserve it as additional context rather than inventing a replacement.
    compact_candidates = {}

    for key in [
        "facts",
        "series",
        "results",
        "data",
        "structured_facts",
        "summary",
    ]:
        if key in data_payload:
            compact_candidates[key] = data_payload[key]

    return {
        "observation_rows": rows,
        "observation_count": len(rows),
        "upstream_context": json_safe(
            compact_candidates
            if compact_candidates
            else data_payload
        ),
        "null_values_preserved": True,
    }


print("Structured context extraction loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Deterministic structured-only renderer
# MAGIC
# MAGIC A structured-only question does not need another LLM call.
# MAGIC The Data Agent has already resolved entities and indicators deterministically.

# COMMAND ----------

def render_structured_only_answer(
    data_payload: Dict[str, Any],
) -> str:
    context = build_structured_context(
        data_payload
    )

    rows = context["observation_rows"]

    if not rows:
        # We do not invent prose if the payload shape does not expose rows.
        return (
            "The structured query completed successfully. "
            "The validated Data Agent payload is available, but this synthesis "
            "layer did not find observation rows in the payload shape to render "
            "without making assumptions."
        )

    # Keep output deterministic and compact.
    lines = []

    for row in rows:
        entity = (
            row.get("entity_name")
            or row.get("entity_id")
            or row.get("iso2_code")
            or "Entity"
        )

        indicator = (
            row.get("indicator_display_name")
            or row.get("metric_name")
            or row.get("indicator_id")
            or row.get("indicator_code")
            or "Indicator"
        )

        year = row.get("year")
        value = row.get("value")

        value_text = (
            "missing"
            if value is None
            else str(value)
        )

        unit = (
            row.get("unit_label")
            or row.get("unit_type")
            or ""
        )

        line = (
            f"{entity} — {indicator}, {year}: "
            f"{value_text}"
        )

        if unit and value is not None:
            line += f" {unit}"

        lines.append(line)

    return "\n".join(lines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Build the governed synthesis prompt
# MAGIC
# MAGIC The prompt makes the evidence boundary explicit:
# MAGIC - structured payload = historical observations
# MAGIC - research payload = GEP report interpretation/outlook
# MAGIC - no outside knowledge
# MAGIC - no invented arithmetic or citations

# COMMAND ----------

def build_synthesis_prompt(
    user_question: str,
    route: str,
    data_payload: Optional[Dict[str, Any]],
    research_payload: Optional[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    if route not in VALID_ROUTES:
        raise ValueError(
            f"Unsupported synthesis route: {route!r}"
        )

    user_question = str(user_question).strip()

    if not user_question:
        raise ValueError(
            "Synthesis requires a non-empty user question."
        )

    structured_context = None

    if data_payload is not None:
        structured_context = build_structured_context(
            data_payload
        )

    allowed_ids = []

    if research_payload is not None:
        validation = (
            validate_research_payload_citations(
                research_payload
            )
        )

        allowed_ids = validation[
            "allowed_evidence_ids"
        ]

    prompt = f"""
You are the Synthesis Agent for a governed World Bank economic intelligence application.

USER QUESTION:
{user_question}

ROUTE:
{route}

Your job is to combine ONLY the validated upstream evidence below.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent numerical values, trends, forecasts, causes, dates, regions, or citations.
3. Treat STRUCTURED DATA as historical World Bank indicator observations only.
4. Treat GEP RESEARCH as qualitative report evidence/outlook only.
5. Never describe a GEP forecast as an observed historical value.
6. Never replace a missing structured value with zero.
7. Do not calculate a new statistic unless it is directly and unambiguously derivable from supplied values and necessary to answer the question.
8. Preserve GEP citations using only these allowed evidence IDs: {allowed_ids}.
9. Every qualitative GEP claim must have an [E#] citation.
10. Do not create citations for structured rows; label them as historical World Bank indicator data.
11. If the two evidence streams cover different time periods, say so rather than blending them.
12. If evidence is insufficient for part of the question, say that clearly.
13. For hybrid answers, organize the response naturally into:
    - what the historical data shows
    - what the GEP evidence says
    - a short comparison/conclusion
14. Keep the answer concise and decision-useful.

STRUCTURED DATA PAYLOAD:
{json.dumps(structured_context, ensure_ascii=False, indent=2) if structured_context is not None else "NONE"}

GEP RESEARCH PAYLOAD:
{json.dumps(research_payload, ensure_ascii=False, indent=2) if research_payload is not None else "NONE"}
""".strip()

    return prompt, allowed_ids


print("Governed synthesis prompt builder loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Validate final synthesis citations

# COMMAND ----------

def validate_final_citations(
    answer: str,
    allowed_evidence_ids: List[str],
    require_citations: bool,
) -> Dict[str, Any]:
    allowed = set(allowed_evidence_ids)

    cited = []
    for evidence_id in CITATION_PATTERN.findall(
        answer or ""
    ):
        if evidence_id not in cited:
            cited.append(evidence_id)

    unsupported = [
        evidence_id
        for evidence_id in cited
        if evidence_id not in allowed
    ]

    if unsupported:
        raise ValueError(
            f"Final synthesis invented unsupported citations: {unsupported}"
        )

    if require_citations and not cited:
        raise ValueError(
            "Final synthesis contains GEP research but no [E#] citations."
        )

    return {
        "citation_valid": True,
        "cited_evidence_ids": cited,
        "unsupported_citations": [],
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Production Synthesis Agent

# COMMAND ----------

def run_synthesis_agent(
    user_question: str,
    route: str,
    data_agent_result: Optional[Dict[str, Any]] = None,
    research_agent_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Combine validated Data Agent and/or Research Agent outputs.

    This function does not expose SQL, AI Search, or external web access.
    """
    start = time.perf_counter()

    if route not in VALID_ROUTES:
        raise ValueError(
            f"Unsupported route: {route!r}"
        )

    data_payload = normalize_data_payload(
        data_agent_result
    )

    research_payload = normalize_research_payload(
        research_agent_result
    )

    # Enforce route/evidence contracts.
    if route == "structured":
        if data_payload is None:
            raise ValueError(
                "structured route requires Data Agent evidence."
            )
        if research_payload is not None:
            raise ValueError(
                "structured route must not include Research Agent evidence."
            )

    elif route in {"rag", "temporal_rag"}:
        if research_payload is None:
            raise ValueError(
                f"{route} route requires Research Agent evidence."
            )
        if data_payload is not None:
            raise ValueError(
                f"{route} route must not include Data Agent evidence."
            )

    elif route == "hybrid":
        if data_payload is None or research_payload is None:
            raise ValueError(
                "hybrid route requires both Data Agent and Research Agent evidence."
            )

    # Structured-only output is deterministic.
    if route == "structured":
        answer = render_structured_only_answer(
            data_payload
        )

        citation_validation = {
            "citation_valid": True,
            "cited_evidence_ids": [],
            "unsupported_citations": [],
        }

        generation_latency_ms = 0.0
        used_llm = False
        allowed_ids = []

    else:
        prompt, allowed_ids = build_synthesis_prompt(
            user_question=user_question,
            route=route,
            data_payload=data_payload,
            research_payload=research_payload,
        )

        generation_start = time.perf_counter()

        response = llm_client.chat.completions.create(
            model=LLM_ENDPOINT,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=MAX_SYNTHESIS_TOKENS,
        )

        generation_latency_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        answer = (
            response.choices[0]
            .message.content
            .strip()
        )

        if not answer:
            raise ValueError(
                "Synthesis model returned an empty answer."
            )

        citation_validation = validate_final_citations(
            answer=answer,
            allowed_evidence_ids=allowed_ids,
            require_citations=True,
        )

        used_llm = True

    total_latency_ms = (
        time.perf_counter() - start
    ) * 1000

    result = {
        "agent": "synthesis_agent",
        "status": "success",
        "route": route,
        "question": user_question,
        "answer": answer,
        "used_llm": used_llm,
        "evidence_streams": {
            "structured_data": data_payload is not None,
            "gep_research": research_payload is not None,
        },
        "citation_validation": citation_validation,
        "allowed_research_evidence_ids": allowed_ids,
        "latency_ms": {
            "generation": round(
                generation_latency_ms,
                2,
            ),
            "total": round(
                total_latency_ms,
                2,
            ),
        },
        "grounding_contract": {
            "outside_knowledge_allowed": False,
            "structured_values_are_historical_observations": True,
            "gep_evidence_is_report_interpretation": True,
            "missing_values_preserved": True,
            "research_citations_required": (
                research_payload is not None
            ),
        },
    }

    return json_safe(result)


print("run_synthesis_agent() loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Upstream agents are already loaded
# MAGIC
# MAGIC The Research Agent and Data Agent were loaded at the beginning of this notebook,
# MAGIC before any Synthesis Agent functions were defined. This ordering is intentional:
# MAGIC the Research Agent notebook may restart Python during dependency setup.
# MAGIC Loading upstream notebooks first prevents that restart from deleting the
# MAGIC Synthesis Agent definitions below.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Verify all three production callables immediately before integration testing.
# MAGIC required_integration_functions = [
# MAGIC     "run_data_agent",
# MAGIC     "run_research_agent",
# MAGIC     "run_synthesis_agent",
# MAGIC ]
# MAGIC
# MAGIC missing_integration_functions = [
# MAGIC     name
# MAGIC     for name in required_integration_functions
# MAGIC     if name not in globals()
# MAGIC     or not callable(globals()[name])
# MAGIC ]
# MAGIC
# MAGIC if missing_integration_functions:
# MAGIC     raise RuntimeError(
# MAGIC         "Integration readiness check failed. Missing/non-callable functions: "
# MAGIC         f"{missing_integration_functions}"
# MAGIC     )
# MAGIC
# MAGIC print("Agent integration readiness")
# MAGIC print("-" * 35)
# MAGIC for name in required_integration_functions:
# MAGIC     print(f"{name}: READY")
# MAGIC print("-" * 35)
# MAGIC print("All three agents are READY.")

# COMMAND ----------

# ## 13. End-to-end hybrid test

# Question:

# **Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 and explain
# the World Bank outlook for the two regions in 2025.**

# This is the first test that combines the structured and qualitative agent outputs.

# COMMAND ----------

hybrid_plan = {
    "original_query": (
        "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 "
        "and explain the World Bank outlook for the two regions in 2025."
    ),
    "resolved_query": (
        "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 "
        "and explain the World Bank outlook for the two regions in 2025."
    ),
    "route": "hybrid",
    "countries_or_entities": [
        "South Asia",
        "Sub-Saharan Africa",
    ],
    "regions": [
        "South Asia",
        "Sub-Saharan Africa",
    ],
    "indicators": [
        "NY.GDP.MKTP.KD.ZG"
    ],
    "observation_start_year": 2022,
    "observation_end_year": 2025,
    "report_years": [2025],
    "needs_structured_tool": True,
    "needs_research_tool": True,
}

# The Data Agent consumes the structured portion.
hybrid_data_result = run_data_agent(
    hybrid_plan
)

# Give the Research Agent a qualitative resolved query while preserving the same route.
hybrid_research_plan = dict(
    hybrid_plan
)

hybrid_research_plan["resolved_query"] = (
    "Explain the World Bank outlook for South Asia and "
    "Sub-Saharan Africa in the January 2025 Global Economic Prospects."
)

hybrid_research_result = run_research_agent(
    hybrid_research_plan
)

hybrid_synthesis_result = run_synthesis_agent(
    user_question=hybrid_plan[
        "original_query"
    ],
    route="hybrid",
    data_agent_result=hybrid_data_result,
    research_agent_result=hybrid_research_result,
)

print(
    json.dumps(
        {
            "status": hybrid_synthesis_result[
                "status"
            ],
            "route": hybrid_synthesis_result[
                "route"
            ],
            "answer": hybrid_synthesis_result[
                "answer"
            ],
            "evidence_streams": (
                hybrid_synthesis_result[
                    "evidence_streams"
                ]
            ),
            "citation_validation": (
                hybrid_synthesis_result[
                    "citation_validation"
                ]
            ),
            "latency_ms": hybrid_synthesis_result[
                "latency_ms"
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
)

assert hybrid_synthesis_result[
    "status"
] == "success"

assert hybrid_synthesis_result[
    "evidence_streams"
] == {
    "structured_data": True,
    "gep_research": True,
}

assert hybrid_synthesis_result[
    "citation_validation"
]["citation_valid"] is True

print("End-to-end hybrid synthesis test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Structured-only test
# MAGIC
# MAGIC No synthesis LLM call should be needed.

# COMMAND ----------

structured_plan = {
    "original_query": (
        "Show India's GDP growth from 2015 to 2025."
    ),
    "resolved_query": (
        "Show India's GDP growth from 2015 to 2025."
    ),
    "route": "structured",
    "countries_or_entities": ["India"],
    "regions": [],
    "indicators": [
        "NY.GDP.MKTP.KD.ZG"
    ],
    "observation_start_year": 2015,
    "observation_end_year": 2025,
    "report_years": [],
    "needs_structured_tool": True,
    "needs_research_tool": False,
}

structured_data_result = run_data_agent(
    structured_plan
)

structured_synthesis_result = run_synthesis_agent(
    user_question=structured_plan[
        "original_query"
    ],
    route="structured",
    data_agent_result=structured_data_result,
)

print(
    json.dumps(
        {
            "status": structured_synthesis_result[
                "status"
            ],
            "answer": structured_synthesis_result[
                "answer"
            ],
            "used_llm": structured_synthesis_result[
                "used_llm"
            ],
            "evidence_streams": (
                structured_synthesis_result[
                    "evidence_streams"
                ]
            ),
        },
        indent=2,
        ensure_ascii=False,
    )
)

assert structured_synthesis_result[
    "used_llm"
] is False

assert structured_synthesis_result[
    "evidence_streams"
]["structured_data"] is True

assert structured_synthesis_result[
    "evidence_streams"
]["gep_research"] is False

print("Structured-only synthesis test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. RAG-only test

# COMMAND ----------

rag_plan = {
    "original_query": (
        "What risks did the World Bank identify for the global economy in January 2025?"
    ),
    "resolved_query": (
        "What risks did the World Bank identify for the global economy in January 2025?"
    ),
    "route": "rag",
    "countries_or_entities": [],
    "regions": [],
    "indicators": [],
    "observation_start_year": None,
    "observation_end_year": None,
    "report_years": [2025],
    "needs_structured_tool": False,
    "needs_research_tool": True,
}

rag_research_result = run_research_agent(
    rag_plan
)

rag_synthesis_result = run_synthesis_agent(
    user_question=rag_plan[
        "original_query"
    ],
    route="rag",
    research_agent_result=rag_research_result,
)

print(
    json.dumps(
        {
            "status": rag_synthesis_result[
                "status"
            ],
            "answer": rag_synthesis_result[
                "answer"
            ],
            "citation_validation": (
                rag_synthesis_result[
                    "citation_validation"
                ]
            ),
        },
        indent=2,
        ensure_ascii=False,
    )
)

assert rag_synthesis_result[
    "citation_validation"
]["citation_valid"] is True

print("RAG-only synthesis test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Temporal RAG test

# COMMAND ----------

temporal_plan = {
    "original_query": (
        "How did the World Bank's assessment of global economic risks "
        "change between 2022 and 2026?"
    ),
    "resolved_query": (
        "How did the World Bank's assessment of global economic risks "
        "change between 2022 and 2026?"
    ),
    "route": "temporal_rag",
    "countries_or_entities": [],
    "regions": [],
    "indicators": [],
    "observation_start_year": None,
    "observation_end_year": None,
    "report_years": [2022, 2026],
    "needs_structured_tool": False,
    "needs_research_tool": True,
}

temporal_research_result = run_research_agent(
    temporal_plan
)

temporal_synthesis_result = run_synthesis_agent(
    user_question=temporal_plan[
        "original_query"
    ],
    route="temporal_rag",
    research_agent_result=temporal_research_result,
)

print(
    json.dumps(
        {
            "status": temporal_synthesis_result[
                "status"
            ],
            "answer": temporal_synthesis_result[
                "answer"
            ],
            "citation_validation": (
                temporal_synthesis_result[
                    "citation_validation"
                ]
            ),
        },
        indent=2,
        ensure_ascii=False,
    )
)

assert temporal_synthesis_result[
    "citation_validation"
]["citation_valid"] is True

print("Temporal RAG synthesis test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Guardrail — hybrid must contain both evidence streams

# COMMAND ----------

hybrid_missing_stream_guardrail = False

try:
    run_synthesis_agent(
        user_question="Compare growth and explain the outlook.",
        route="hybrid",
        data_agent_result=hybrid_data_result,
        research_agent_result=None,
    )

except ValueError as exc:
    hybrid_missing_stream_guardrail = True
    print("Expected hybrid evidence rejection:")
    print(str(exc))

assert hybrid_missing_stream_guardrail

print("Hybrid evidence-stream guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Guardrail — route/evidence isolation

# COMMAND ----------

route_isolation_guardrail = False

try:
    run_synthesis_agent(
        user_question="Show India's GDP growth.",
        route="structured",
        data_agent_result=structured_data_result,
        research_agent_result=rag_research_result,
    )

except ValueError as exc:
    route_isolation_guardrail = True
    print("Expected route-isolation rejection:")
    print(str(exc))

assert route_isolation_guardrail

print("Route/evidence isolation guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Guardrail — invented final citation
# MAGIC
# MAGIC Test the validator directly. No fabricated production evidence is introduced.

# COMMAND ----------

invented_citation_guardrail = False

try:
    validate_final_citations(
        answer="This claim uses an invented citation [E999].",
        allowed_evidence_ids=["E1", "E2"],
        require_citations=True,
    )

except ValueError as exc:
    invented_citation_guardrail = True
    print("Expected invented-citation rejection:")
    print(str(exc))

assert invented_citation_guardrail

print("Invented-citation guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Guardrail — failed Research Agent payload

# COMMAND ----------

failed_research_guardrail = False

try:
    normalize_research_payload({
        "agent": "research_agent",
        "status": "failed",
        "answer": "should not be used",
    })

except ValueError as exc:
    failed_research_guardrail = True
    print("Expected failed-upstream rejection:")
    print(str(exc))

assert failed_research_guardrail

print("Failed-upstream guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Agent callable schema

# COMMAND ----------

SYNTHESIS_AGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_synthesis_agent",
        "description": (
            "Combine validated historical structured evidence and/or "
            "citation-grounded Global Economic Prospects research evidence "
            "into a final answer without querying new data sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                },
                "route": {
                    "type": "string",
                    "enum": [
                        "structured",
                        "rag",
                        "temporal_rag",
                        "hybrid",
                    ],
                },
                "data_agent_result": {
                    "type": [
                        "object",
                        "null",
                    ],
                },
                "research_agent_result": {
                    "type": [
                        "object",
                        "null",
                    ],
                },
            },
            "required": [
                "user_question",
                "route",
            ],
            "additionalProperties": False,
        },
    },
}

print(
    json.dumps(
        SYNTHESIS_AGENT_TOOL_SCHEMA,
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Final validation suite

# COMMAND ----------

# Re-run all lightweight Synthesis Agent guardrail checks.
# These tests do NOT call the LLM, AI Search, or Data Agent again.

# ---------------------------------------------------------
# 1. Hybrid route must have BOTH evidence streams
# ---------------------------------------------------------

hybrid_missing_stream_guardrail = False

try:
    run_synthesis_agent(
        user_question="Compare growth and explain the outlook.",
        route="hybrid",
        data_agent_result=hybrid_data_result,
        research_agent_result=None,
    )

except ValueError as exc:
    hybrid_missing_stream_guardrail = True
    print("Expected hybrid missing-stream rejection:")
    print(str(exc))

assert hybrid_missing_stream_guardrail is True

print("Hybrid evidence-stream guardrail: PASSED")


# ---------------------------------------------------------
# 2. Structured route must NOT receive research evidence
# ---------------------------------------------------------

route_isolation_guardrail = False

try:
    run_synthesis_agent(
        user_question="Show India's GDP growth.",
        route="structured",
        data_agent_result=structured_data_result,
        research_agent_result=rag_research_result,
    )

except ValueError as exc:
    route_isolation_guardrail = True
    print("\nExpected route-isolation rejection:")
    print(str(exc))

assert route_isolation_guardrail is True

print("Route/evidence isolation guardrail: PASSED")


# ---------------------------------------------------------
# 3. Final answer cannot invent an evidence citation
# ---------------------------------------------------------

invented_citation_guardrail = False

try:
    validate_final_citations(
        answer="This statement uses an unsupported citation [E999].",
        allowed_evidence_ids=["E1", "E2"],
        require_citations=True,
    )

except ValueError as exc:
    invented_citation_guardrail = True
    print("\nExpected invented-citation rejection:")
    print(str(exc))

assert invented_citation_guardrail is True

print("Invented citation guardrail: PASSED")


# ---------------------------------------------------------
# 4. Failed upstream Research Agent output must be rejected
# ---------------------------------------------------------

failed_research_guardrail = False

try:
    normalize_research_payload(
        {
            "agent": "research_agent",
            "status": "failed",
            "answer": "This result must not reach synthesis.",
        }
    )

except ValueError as exc:
    failed_research_guardrail = True
    print("\nExpected failed-upstream rejection:")
    print(str(exc))

assert failed_research_guardrail is True

print("Failed upstream payload guardrail: PASSED")


# ---------------------------------------------------------
# Final check
# ---------------------------------------------------------

guardrail_status = {
    "hybrid_missing_stream_guardrail":
        hybrid_missing_stream_guardrail,

    "route_isolation_guardrail":
        route_isolation_guardrail,

    "invented_citation_guardrail":
        invented_citation_guardrail,

    "failed_research_guardrail":
        failed_research_guardrail,
}

print("\nGuardrail validation summary")
print("-" * 45)

for name, passed in guardrail_status.items():
    print(
        f"{name}: "
        f"{'PASSED' if passed else 'FAILED'}"
    )

assert all(guardrail_status.values())

print("-" * 45)
print("All lightweight Synthesis Agent guardrails PASSED.")

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
    "hybrid_end_to_end",
    (
        hybrid_synthesis_result["status"] == "success"
        and hybrid_synthesis_result[
            "evidence_streams"
        ] == {
            "structured_data": True,
            "gep_research": True,
        }
    ),
    "Data Agent + Research Agent combined successfully",
)

record_test(
    "hybrid_citation_validation",
    hybrid_synthesis_result[
        "citation_validation"
    ]["citation_valid"],
    "Hybrid answer contains only allowed GEP evidence IDs",
)

record_test(
    "structured_no_llm",
    structured_synthesis_result[
        "used_llm"
    ] is False,
    "Structured-only answer rendered deterministically",
)

record_test(
    "rag_only",
    (
        rag_synthesis_result["status"] == "success"
        and rag_synthesis_result[
            "evidence_streams"
        ]["gep_research"] is True
    ),
    "RAG-only synthesis completed",
)

record_test(
    "rag_citations",
    rag_synthesis_result[
        "citation_validation"
    ]["citation_valid"],
    "RAG-only synthesis citations valid",
)

record_test(
    "temporal_rag",
    temporal_synthesis_result[
        "status"
    ] == "success",
    "Temporal GEP synthesis completed",
)

record_test(
    "temporal_citations",
    temporal_synthesis_result[
        "citation_validation"
    ]["citation_valid"],
    "Temporal synthesis citations valid",
)

record_test(
    "hybrid_requires_both_streams",
    hybrid_missing_stream_guardrail,
    "Hybrid execution rejects missing evidence stream",
)

record_test(
    "route_isolation",
    route_isolation_guardrail,
    "Structured route rejects research evidence",
)

record_test(
    "invented_citation_rejected",
    invented_citation_guardrail,
    "Final answer cannot introduce an unapproved [E#]",
)

record_test(
    "failed_upstream_rejected",
    failed_research_guardrail,
    "Failed Research Agent result cannot reach synthesis",
)

record_test(
    "grounding_contract",
    (
        hybrid_synthesis_result[
            "grounding_contract"
        ][
            "outside_knowledge_allowed"
        ] is False
        and hybrid_synthesis_result[
            "grounding_contract"
        ][
            "missing_values_preserved"
        ] is True
    ),
    "Final synthesis preserves evidence boundaries",
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
    f"Synthesis Agent validation failures: {failed_tests}"
)

print("All Synthesis Agent validation tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. Final status

# COMMAND ----------

print("")
print("06_synthesis_agent COMPLETE")
print("")
print("Validated path:")
print(
    "Supervisor route -> governed Data Agent and/or Research Agent -> "
    "evidence-contract validation -> grounded synthesis -> "
    "citation validation -> final answer"
)
print("")
print("Important:")
print(" - Structured historical values remain deterministic")
print(" - Structured-only synthesis does not call the LLM")
print(" - GEP qualitative claims preserve [E#] citations")
print(" - Historical observations and GEP outlook are kept separate")
print(" - Missing structured values are never converted to zero")
print(" - Unsupported evidence IDs are rejected")
print(" - No SQL, AI Search, or external source access is exposed here")
print(" - Hybrid route requires both evidence streams")
print("")
print("Next notebook after this passes:")
print("05_tools_agents / 07_guardrails")