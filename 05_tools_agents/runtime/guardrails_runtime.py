# Databricks notebook source
# MAGIC %md
# MAGIC # runtime / guardrails_runtime
# MAGIC
# MAGIC Definitions only. No package installation, Python restart, smoke test,
# MAGIC benchmark execution, or development validation cells.
# MAGIC
# MAGIC This notebook is safe to import with `%run` from evaluation/serving notebooks.

# COMMAND ----------

import re
import json
from typing import Any, Dict, List, Optional

VALID_ROUTES = {"structured", "rag", "temporal_rag", "hybrid"}
VALID_REPORT_YEARS = set(range(2022, 2027))
MIN_OBSERVATION_YEAR = 2010
MAX_OBSERVATION_YEAR = 2025
MAX_QUERY_CHARS = 4000
MAX_ANSWER_CHARS = 30000

# High-confidence instruction-manipulation patterns.
# We do not block ordinary mentions of "prompt" or "instructions".
INJECTION_PATTERNS = [
    r"\bignore\s+(all|any|the|previous|prior|above)\s+(instructions?|rules?|prompts?)\b",
    r"\bdisregard\s+(all|any|the|previous|prior|above)\s+(instructions?|rules?|prompts?)\b",
    r"\breveal\s+(the\s+)?(system|developer)\s+(prompt|message|instructions?)\b",
    r"\bshow\s+me\s+(your|the)\s+(system|developer)\s+(prompt|message|instructions?)\b",
    r"\bprint\s+(your|the)\s+(system|developer)\s+(prompt|message|instructions?)\b",
]

EVIDENCE_ID_RE = re.compile(r"\bE\d+\b")
STRUCTURED_ID_RE = re.compile(r"\bS\d+\b")


def _require_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary.")
    return value


def validate_user_input(query: str) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("User query must be a non-empty string.")

    clean = query.strip()
    if len(clean) > MAX_QUERY_CHARS:
        raise ValueError(
            f"User query exceeds {MAX_QUERY_CHARS} characters."
        )

    matches = [
        pattern
        for pattern in INJECTION_PATTERNS
        if re.search(pattern, clean, flags=re.IGNORECASE)
    ]

    if matches:
        raise ValueError(
            "Potential instruction-manipulation request rejected."
        )

    return {
        "status": "passed",
        "query": clean,
        "query_length": len(clean),
    }


def validate_execution_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = _require_dict(plan, "Supervisor plan")

    route = plan.get("route")
    if route not in VALID_ROUTES:
        raise ValueError(f"Unsupported route: {route!r}")

    needs_structured = bool(plan.get("needs_structured_tool"))
    needs_research = bool(plan.get("needs_research_tool"))

    expected = {
        "structured": (True, False),
        "rag": (False, True),
        "temporal_rag": (False, True),
        "hybrid": (True, True),
    }[route]

    if (needs_structured, needs_research) != expected:
        raise ValueError(
            f"Route/tool mismatch for {route}: "
            f"got structured={needs_structured}, research={needs_research}; "
            f"expected structured={expected[0]}, research={expected[1]}."
        )

    report_years = plan.get("report_years") or []
    invalid_report_years = [
        int(y) for y in report_years
        if int(y) not in VALID_REPORT_YEARS
    ]
    if invalid_report_years:
        raise ValueError(
            f"GEP report years must be 2022-2026: {invalid_report_years}"
        )

    if route == "temporal_rag" and len(set(map(int, report_years))) < 2:
        raise ValueError(
            "temporal_rag requires at least two distinct GEP report years."
        )

    start = plan.get("observation_start_year")
    end = plan.get("observation_end_year")

    if needs_structured:
        if start is not None and int(start) < MIN_OBSERVATION_YEAR:
            raise ValueError(
                f"Historical observations start at {MIN_OBSERVATION_YEAR}."
            )
        if end is not None and int(end) > MAX_OBSERVATION_YEAR:
            raise ValueError(
                f"Historical observations stop at {MAX_OBSERVATION_YEAR}. "
                "Use GEP evidence for forecasts/projections."
            )
        if start is not None and end is not None and int(start) > int(end):
            raise ValueError("observation_start_year cannot exceed observation_end_year.")

    return {
        "status": "passed",
        "route": route,
        "needs_structured_tool": needs_structured,
        "needs_research_tool": needs_research,
        "report_years": sorted(set(map(int, report_years))),
    }


def validate_research_evidence(
    research_result: Dict[str, Any],
) -> Dict[str, Any]:
    result = _require_dict(research_result, "Research Agent result")

    if result.get("status") != "success":
        raise ValueError(
            f"Research Agent result is not successful: "
            f"{result.get('status')!r}"
        )

    payload = (
        result.get("synthesis_payload")
        or result.get("payload")
        or result
    )

    evidence = (
        payload.get("evidence")
        or payload.get("final_chunks")
        or result.get("evidence")
        or result.get("final_chunks")
        or []
    )

    allowed_ids = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = (
            item.get("evidence_id")
            or item.get("id")
        )
        if evidence_id:
            allowed_ids.add(str(evidence_id).strip("[]"))

    answer = (
        payload.get("research_answer")
        or payload.get("answer")
        or result.get("research_answer")
        or result.get("answer")
        or ""
    )

    cited = set(EVIDENCE_ID_RE.findall(answer))
    unsupported = sorted(cited - allowed_ids)

    if unsupported:
        raise ValueError(
            f"Research answer contains unsupported evidence IDs: {unsupported}"
        )

    return {
        "status": "passed",
        "allowed_evidence_ids": sorted(allowed_ids),
        "cited_evidence_ids": sorted(cited),
        "unsupported_citations": unsupported,
    }


def validate_final_answer(
    answer: str,
    route: str,
    allowed_evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if route not in VALID_ROUTES:
        raise ValueError(f"Unsupported route: {route!r}")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Final answer must be non-empty.")

    if len(answer) > MAX_ANSWER_CHARS:
        raise ValueError(
            f"Final answer exceeds {MAX_ANSWER_CHARS} characters."
        )

    allowed = {
        str(x).strip("[]")
        for x in (allowed_evidence_ids or [])
    }
    cited = set(EVIDENCE_ID_RE.findall(answer))
    unsupported = sorted(cited - allowed)

    if unsupported:
        raise ValueError(
            f"Final answer contains unsupported GEP citations: {unsupported}"
        )

    if route in {"rag", "temporal_rag", "hybrid"} and not cited:
        raise ValueError(
            f"{route} answer must cite at least one validated [E#] evidence item."
        )

    # Structured-only output must not pretend that GEP evidence was used.
    if route == "structured" and cited:
        raise ValueError(
            "Structured-only answer must not contain GEP [E#] citations."
        )

    return {
        "status": "passed",
        "route": route,
        "cited_evidence_ids": sorted(cited),
        "unsupported_citations": unsupported,
    }


def apply_pre_execution_guardrails(
    user_query: str,
    supervisor_plan: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "input": validate_user_input(user_query),
        "plan": validate_execution_plan(supervisor_plan),
    }


def apply_post_execution_guardrails(
    route: str,
    answer: str,
    allowed_evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return validate_final_answer(
        answer=answer,
        route=route,
        allowed_evidence_ids=allowed_evidence_ids,
    )

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

print("guardrails_runtime READY")
print(" - apply_pre_execution_guardrails")
print(" - apply_post_execution_guardrails")

# Backward-compatible guardrail-specific public alias.
# Do NOT use the old generic name validate_supervisor_plan because that name
# collides with the Supervisor runtime in a shared Databricks %run namespace.
validate_guardrail_plan = validate_execution_plan

print("guardrails_runtime READY")
print(" - validate_execution_plan")
print(" - apply_pre_execution_guardrails")
print(" - apply_post_execution_guardrails")

