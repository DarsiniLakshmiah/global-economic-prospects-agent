# Databricks notebook source
# MAGIC %md
# MAGIC # runtime / synthesis_agent_runtime
# MAGIC
# MAGIC Definitions only. No package installation, Python restart, smoke test,
# MAGIC benchmark execution, or development validation cells.
# MAGIC
# MAGIC This notebook is safe to import with `%run` from evaluation/serving notebooks.

# COMMAND ----------

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
    inherited: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Recursively collect structured observation rows.

    The compact Data Agent synthesis contract stores entity/indicator metadata
    on each parent series and year/value/unit on child observations. Carry the
    parent metadata into each observation row so structured-only rendering does
    not lose entity or indicator provenance.
    """
    rows = []
    inherited = dict(inherited or {})

    if isinstance(obj, dict):
        current = dict(inherited)

        # Propagate only descriptive series metadata. Never overwrite the
        # observation's own year/value/unit fields.
        for key in [
            "entity_id",
            "entity_name",
            "entity_type",
            "iso2_code",
            "indicator_id",
            "indicator_code",
            "indicator_name",
            "metric_name",
            "indicator_display_name",
            "unit_type",
            "unit_label",
        ]:
            value = obj.get(key)
            if value is not None:
                current[key] = value

        if "year" in obj and "value" in obj:
            row = dict(current)
            row.update(obj)

            # indicator_name is the compact Data Agent field. Normalize it to
            # the renderer's preferred display field without changing content.
            if (
                row.get("indicator_display_name") is None
                and row.get("indicator_name") is not None
            ):
                row["indicator_display_name"] = row["indicator_name"]

            rows.append(json_safe(row))
        else:
            for value in obj.values():
                rows.extend(
                    collect_observation_rows(
                        value,
                        inherited=current,
                    )
                )

    elif isinstance(obj, list):
        for item in obj:
            rows.extend(
                collect_observation_rows(
                    item,
                    inherited=inherited,
                )
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

# COMMAND ----------

print("synthesis_agent_runtime READY")
print(" - run_synthesis_agent")
