# ============================================================
# World Bank GEP Intelligence Agent
# Production Orchestrator
# ============================================================

from typing import Any, Dict, List, Optional

from serving_runtime.supervisor_runtime import (
    run_supervisor_agent,
)

from serving_runtime.data_agent_sql_runtime import (
    run_data_agent,
)

from serving_runtime.research_agent_runtime import (
    run_research_agent,
)

from serving_runtime.synthesis_agent_runtime import (
    run_synthesis_agent,
)

from serving_runtime.guardrails_runtime import (
    apply_pre_execution_guardrails,
    apply_post_execution_guardrails,
)

def _extract_allowed_evidence_ids(research_result):
    if not research_result:
        return []
    package = research_result.get("context_package") or {}
    return package.get("allowed_evidence_ids") or []

def execute_agent(
    question: str,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()

    plan = run_supervisor_agent(
        question,
        conversation_context=conversation_context,
    )

    guarded = apply_pre_execution_guardrails(plan)
    plan = guarded.get("plan", plan)

    route = plan["route"]
    data_result = None
    research_result = None

    if route in {"structured", "hybrid"}:
        data_result = run_data_agent(plan)

    if route in {"rag", "temporal_rag", "hybrid"}:
        research_result = run_research_agent(plan)

    synthesis = run_synthesis_agent(
        user_question=question,
        route=route,
        data_agent_result=data_result,
        research_agent_result=research_result,
    )

    allowed_ids = _extract_allowed_evidence_ids(research_result)

    post = apply_post_execution_guardrails(
        route=route,
        answer=synthesis["answer"],
        allowed_evidence_ids=allowed_ids,
    )

    return {
        "status": synthesis.get("status", "success"),
        "question": question,
        "route": route,
        "plan": plan,
        "answer": synthesis["answer"],
        "citation_validation": synthesis.get("citation_validation"),
        "research": research_result,
        "structured": data_result,
        "guardrails": {
            "pre": guarded,
            "post": post,
        },
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
