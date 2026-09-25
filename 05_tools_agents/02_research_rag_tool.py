# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 02_research_rag_tool
# MAGIC
# MAGIC Agent-safe research tool over the locked GEP RAG stack.
# MAGIC
# MAGIC This notebook does **not** rebuild embeddings, indexes, chunking, or the baseline RAG system.
# MAGIC It wraps the validated production candidate:
# MAGIC
# MAGIC `query -> HYBRID AI Search -> report-year filter -> optional LLM rerank -> evidence dedupe -> token budget -> citation-ready evidence`
# MAGIC
# MAGIC The tool returns evidence only. Final answer synthesis belongs to the later Synthesis Agent.
# MAGIC
# MAGIC ### Locked resources reused
# MAGIC - AI Search endpoint: `worldbank-gep-ai-search`
# MAGIC - Index: `worldbank_ai.rag.gep_structure_v1_qwen3_index`
# MAGIC - LLM reranker: `databricks-meta-llama-3-3-70b-instruct`
# MAGIC - Child retrieval + parent provenance
# MAGIC - Hybrid retrieval
# MAGIC - Report-year filtering
# MAGIC - Conditional reranking
# MAGIC
# MAGIC ### Important design rule
# MAGIC Numerical historical macroeconomic facts belong to Notebook 01's deterministic
# MAGIC structured tools. This tool is for qualitative GEP evidence, outlooks, risks,
# MAGIC policy discussion, report comparisons, and other document-grounded research.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Install required SDKs
# MAGIC
# MAGIC This uses the exact AI Search SDK path already validated in the RAG notebooks.

# COMMAND ----------

# MAGIC %pip install -q databricks-ai-search databricks-openai

# COMMAND ----------

# Restart once after package installation so imports are deterministic.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Imports and configuration

# COMMAND ----------

from typing import Any, Dict, List, Optional
import json
import math
import re
import time

from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
INDEX_NAME = "worldbank_ai.rag.gep_structure_v1_qwen3_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

VALID_REPORT_YEARS = {2022, 2023, 2024, 2025, 2026}

# Retrieval policy from the locked RAG experiments.
DIRECT_RETRIEVAL_K = 10
RERANK_CANDIDATE_K = 20
FINAL_TOP_K = 6

# Context/evidence policy from the locked context builder.
MAX_EVIDENCE_ITEMS = 6
MAX_CONTEXT_TOKENS = 6000
CHARS_PER_TOKEN = 4.0
MAX_TOKENS_PER_EVIDENCE = 1400
NEAR_DUPLICATE_JACCARD = 0.82

# LLM reranking was useful but expensive in Notebook 06.
# Therefore it is NOT the default for ordinary single-report retrieval.
RERANK_BATCH_SIZE = 10
MAX_TEXT_CHARS_PER_CANDIDATE = 3500

RETURN_COLUMNS = [
    "chunk_id",
    "parent_chunk_id",
    "document_id",
    "report_year",
    "edition_status",
    "chapter",
    "region",
    "section",
    "subsection",
    "content_type",
    "page_start",
    "page_end",
    "chunk_text",
    "retrieval_text",
]

print("Research tool configuration loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Connect to the locked AI Search index and LLM endpoint

# COMMAND ----------

ai_search_client = AISearchClient()

index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
)

llm_client = DatabricksOpenAI()

# Small model-serving health check.
health = llm_client.chat.completions.create(
    model=LLM_ENDPOINT,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly MODEL_OK"
        }
    ],
    temperature=0,
    max_tokens=10,
)

health_text = health.choices[0].message.content.strip()

assert "MODEL_OK" in health_text, (
    f"Model serving health check failed: {health_text}"
)

print("AI Search connection created.")
print("Model serving health check passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Validate report-year input

# COMMAND ----------

def validate_report_years(
    report_years: Optional[List[int]],
) -> List[int]:
    """
    Validate GEP report editions.

    Only January GEP editions loaded into this project are permitted:
    2022, 2023, 2024, 2025, 2026.
    """
    if report_years is None:
        return []

    if not isinstance(report_years, list):
        raise ValueError(
            "report_years must be a list of integers, for example [2025]."
        )

    years = sorted({
        int(year)
        for year in report_years
    })

    invalid = [
        year
        for year in years
        if year not in VALID_REPORT_YEARS
    ]

    if invalid:
        raise ValueError(
            f"Unsupported GEP report year(s): {invalid}. "
            f"Available editions: {sorted(VALID_REPORT_YEARS)}"
        )

    return years


assert validate_report_years([2025]) == [2025]
assert validate_report_years([2026, 2022]) == [2022, 2026]

print("Report-year validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Parse Databricks AI Search responses

# COMMAND ----------

def parse_search_response(
    response: Dict[str, Any],
    columns: List[str],
) -> List[Dict[str, Any]]:
    """
    Convert AI Search's manifest/data_array response into dictionaries.
    """
    if not isinstance(response, dict):
        raise ValueError(
            f"Unexpected AI Search response type: {type(response)}"
        )

    result = response.get("result")

    if not isinstance(result, dict):
        raise ValueError(
            "AI Search response is missing the 'result' object."
        )

    data_array = result.get("data_array", [])

    if data_array is None:
        data_array = []

    parsed = []

    for rank, row in enumerate(data_array, start=1):
        if len(row) < len(columns):
            raise ValueError(
                "AI Search row contains fewer values than requested columns."
            )

        item = dict(zip(columns, row[:len(columns)]))
        item["retrieval_rank"] = rank

        # Databricks may append score-like values after requested columns.
        extra = row[len(columns):]

        if extra:
            item["search_score"] = extra[0]

        parsed.append(item)

    return parsed

print("AI Search response parser loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Build safe AI Search filters
# MAGIC
# MAGIC We filter by report year because that metadata was reliable in the locked RAG evaluation.
# MAGIC We intentionally do not hard-filter region/section because earlier metadata analysis showed
# MAGIC those fields can be null or abbreviated.

# COMMAND ----------

def build_search_filters(
    report_years: List[int],
) -> Optional[Dict[str, Any]]:
    """
    Build Databricks AI Search metadata filters.

    One report year:
        {"report_year": 2025}

    Multiple report years:
        {"report_year": [2022, 2026]}
    """
    years = validate_report_years(report_years)

    if not years:
        return None

    if len(years) == 1:
        return {
            "report_year": years[0]
        }

    return {
        "report_year": years
    }

print(build_search_filters([2025]))
print(build_search_filters([2022, 2026]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Hybrid retrieval
# MAGIC
# MAGIC Databricks `HYBRID` retrieval already combines semantic and lexical retrieval.
# MAGIC We do not add custom BM25/RRF because the locked evaluation did not justify it.

# COMMAND ----------

def hybrid_retrieve(
    query: str,
    report_years: Optional[List[int]] = None,
    num_results: int = DIRECT_RETRIEVAL_K,
) -> Dict[str, Any]:
    """
    Run governed HYBRID retrieval against the locked structure-aware index.
    """
    if not query or not str(query).strip():
        raise ValueError("query cannot be empty.")

    query = str(query).strip()
    years = validate_report_years(report_years or [])

    num_results = int(num_results)

    if num_results < 1 or num_results > 50:
        raise ValueError(
            "num_results must be between 1 and 50."
        )

    filters = build_search_filters(years)

    kwargs = {
        "query_text": query,
        "columns": RETURN_COLUMNS,
        "num_results": num_results,
        "query_type": "HYBRID",
    }

    if filters:
        kwargs["filters"] = filters

    start = time.perf_counter()

    response = index.similarity_search(**kwargs)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    candidates = parse_search_response(
        response=response,
        columns=RETURN_COLUMNS,
    )

    return {
        "query": query,
        "report_years": years,
        "search_filters": filters,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "retrieval_latency_ms": round(latency_ms, 2),
    }

print("Hybrid retrieval function loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Retrieval smoke test — single report

# COMMAND ----------

single_year_test = hybrid_retrieve(
    query="What risks did the World Bank identify for South Asia in 2025?",
    report_years=[2025],
    num_results=6,
)

print("Candidate count:", single_year_test["candidate_count"])
print("Retrieval latency ms:", single_year_test["retrieval_latency_ms"])

assert single_year_test["candidate_count"] > 0

for item in single_year_test["candidates"]:
    assert int(item["report_year"]) == 2025

print("Single-report HYBRID retrieval passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Conditional reranking policy
# MAGIC
# MAGIC Notebook 06 showed only modest quality gains from LLM reranking but very large latency.
# MAGIC Therefore:
# MAGIC - ordinary single-report research: skip reranking by default
# MAGIC - temporal/multi-edition comparison: rerank by default
# MAGIC - caller can explicitly override the decision

# COMMAND ----------

def should_rerank(
    report_years: List[int],
    force_rerank: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Decide whether the expensive LLM reranker should run.
    """
    if force_rerank is not None:
        return {
            "apply": bool(force_rerank),
            "reason": "explicit_override",
        }

    if len(report_years) > 1:
        return {
            "apply": True,
            "reason": "multi_report_temporal_comparison",
        }

    return {
        "apply": False,
        "reason": "single_report_low_latency_default",
    }

print(should_rerank([2025]))
print(should_rerank([2022, 2026]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. LLM relevance reranker
# MAGIC
# MAGIC This reranker only reorders retrieved evidence. It does not generate the final answer.

# COMMAND ----------

def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON array from the reranker response.
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

    if not isinstance(parsed, list):
        raise ValueError(
            "Reranker response must be a JSON array."
        )

    return parsed


def llm_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = FINAL_TOP_K,
) -> Dict[str, Any]:
    """
    Score candidate chunks for relevance using the validated Databricks LLM.

    Candidates are processed in batches to bound prompt size.
    """
    if not candidates:
        return {
            "results": [],
            "rerank_latency_ms": 0.0,
            "rerank_calls": 0,
        }

    start = time.perf_counter()

    scored = []
    calls = 0

    for batch_start in range(
        0,
        len(candidates),
        RERANK_BATCH_SIZE,
    ):
        batch = candidates[
            batch_start:
            batch_start + RERANK_BATCH_SIZE
        ]

        candidate_payload = []

        for local_idx, item in enumerate(batch):
            global_idx = batch_start + local_idx

            text = (
                item.get("retrieval_text")
                or item.get("chunk_text")
                or ""
            )

            candidate_payload.append({
                "candidate_id": global_idx,
                "report_year": item.get("report_year"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "text": text[:MAX_TEXT_CHARS_PER_CANDIDATE],
            })

        prompt = f"""
You are a retrieval relevance reranker for World Bank Global Economic Prospects evidence.

Question:
{query}

Score each candidate from 0 to 100 for how directly it helps answer the question.

Rules:
- Judge relevance only.
- Prefer evidence that directly addresses the requested topic.
- For temporal questions, evidence from each requested report period can be relevant.
- Do not invent facts.
- Return ONLY a JSON array.
- Include every candidate exactly once.
- Use this schema:
[
  {{"candidate_id": 0, "score": 95}},
  {{"candidate_id": 1, "score": 40}}
]

Candidates:
{json.dumps(candidate_payload, ensure_ascii=False)}
""".strip()

        response = llm_client.chat.completions.create(
            model=LLM_ENDPOINT,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=1000,
        )

        calls += 1

        parsed_scores = _extract_json_array(
            response.choices[0].message.content
        )

        score_map = {}

        for row in parsed_scores:
            candidate_id = int(row["candidate_id"])
            score = float(row["score"])

            score_map[candidate_id] = max(
                0.0,
                min(100.0, score),
            )

        for local_idx, item in enumerate(batch):
            global_idx = batch_start + local_idx

            copied = dict(item)
            copied["rerank_score"] = score_map.get(
                global_idx,
                0.0,
            )

            scored.append(copied)

    scored.sort(
        key=lambda item: (
            -float(item.get("rerank_score", 0.0)),
            int(item.get("retrieval_rank", 10**9)),
        )
    )

    for rank, item in enumerate(scored, start=1):
        item["rerank_rank"] = rank

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "results": scored[:top_k],
        "rerank_latency_ms": round(latency_ms, 2),
        "rerank_calls": calls,
    }

print("LLM reranker loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Evidence deduplication helpers

# COMMAND ----------

def normalize_for_overlap(text: str) -> List[str]:
    """Normalize text into tokens for near-duplicate comparison."""
    return re.findall(
        r"[a-z0-9]+",
        (text or "").lower(),
    )


def jaccard_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """Token-set Jaccard similarity."""
    set_a = set(normalize_for_overlap(text_a))
    set_b = set(normalize_for_overlap(text_b))

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate_candidates(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Remove exact chunk duplicates and strong near-overlap duplicates.
    Preserve ranking order.
    """
    kept = []
    seen_chunk_ids = set()

    exact_removed = 0
    near_removed = 0

    for candidate in candidates:
        chunk_id = candidate.get("chunk_id")

        if chunk_id and chunk_id in seen_chunk_ids:
            exact_removed += 1
            continue

        candidate_text = (
            candidate.get("chunk_text")
            or candidate.get("retrieval_text")
            or ""
        )

        is_near_duplicate = False

        for existing in kept:
            existing_text = (
                existing.get("chunk_text")
                or existing.get("retrieval_text")
                or ""
            )

            if (
                jaccard_similarity(
                    candidate_text,
                    existing_text,
                )
                >= NEAR_DUPLICATE_JACCARD
            ):
                is_near_duplicate = True
                break

        if is_near_duplicate:
            near_removed += 1
            continue

        kept.append(candidate)

        if chunk_id:
            seen_chunk_ids.add(chunk_id)

    return {
        "candidates": kept,
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
    }

print("Evidence deduplication helpers loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Token-budgeted citation-ready evidence

# COMMAND ----------

def estimate_tokens(text: str) -> int:
    """
    Deterministic token estimate used by the locked context builder.
    """
    return int(
        math.ceil(
            len(text or "") / CHARS_PER_TOKEN
        )
    )


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
) -> str:
    """
    Deterministic character-based truncation using the same approximation
    as the locked context builder.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = int(
        max_tokens * CHARS_PER_TOKEN
    )

    truncated = text[:max_chars]

    # Prefer ending at a whitespace boundary.
    last_space = truncated.rfind(" ")

    if last_space > int(max_chars * 0.8):
        truncated = truncated[:last_space]

    return truncated.rstrip()


def build_evidence_package(
    query: str,
    candidates: List[Dict[str, Any]],
    report_years: List[int],
) -> Dict[str, Any]:
    """
    Convert ranked chunks into compact citation-ready evidence.

    Evidence IDs E1..En are internal IDs for later synthesis/citation validation.
    """
    dedup = deduplicate_candidates(candidates)

    evidence = []
    total_tokens = 0
    budget_drops = 0

    for candidate in dedup["candidates"]:
        if len(evidence) >= MAX_EVIDENCE_ITEMS:
            break

        raw_text = (
            candidate.get("chunk_text")
            or candidate.get("retrieval_text")
            or ""
        ).strip()

        if not raw_text:
            continue

        evidence_text = truncate_to_token_budget(
            raw_text,
            MAX_TOKENS_PER_EVIDENCE,
        )

        evidence_tokens = estimate_tokens(
            evidence_text
        )

        if (
            total_tokens + evidence_tokens
            > MAX_CONTEXT_TOKENS
        ):
            budget_drops += 1
            continue

        evidence_id = f"E{len(evidence) + 1}"

        evidence.append({
            "evidence_id": evidence_id,
            "chunk_id": candidate.get("chunk_id"),
            "parent_chunk_id": candidate.get("parent_chunk_id"),
            "document_id": candidate.get("document_id"),
            "report_year": candidate.get("report_year"),
            "edition_status": candidate.get("edition_status"),
            "chapter": candidate.get("chapter"),
            "region": candidate.get("region"),
            "section": candidate.get("section"),
            "subsection": candidate.get("subsection"),
            "content_type": candidate.get("content_type"),
            "page_start": candidate.get("page_start"),
            "page_end": candidate.get("page_end"),
            "retrieval_rank": candidate.get("retrieval_rank"),
            "rerank_rank": candidate.get("rerank_rank"),
            "rerank_score": candidate.get("rerank_score"),
            "estimated_tokens": evidence_tokens,
            "text": evidence_text,
        })

        total_tokens += evidence_tokens

    return {
        "query": query,
        "report_years": report_years,
        "evidence": evidence,
        "allowed_evidence_ids": [
            item["evidence_id"]
            for item in evidence
        ],
        "evidence_count": len(evidence),
        "estimated_context_tokens": total_tokens,
        "exact_duplicates_removed": dedup[
            "exact_duplicates_removed"
        ],
        "near_duplicates_removed": dedup[
            "near_duplicates_removed"
        ],
        "budget_drops": budget_drops,
    }

print("Evidence-package builder loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Production research tool
# MAGIC
# MAGIC This is the function the later Research Agent/Supervisor will call.

# COMMAND ----------

def research_gep(
    query: str,
    report_years: Optional[List[int]] = None,
    force_rerank: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Retrieve citation-ready evidence from January GEP editions.

    Parameters
    ----------
    query:
        Natural-language research question.

    report_years:
        Optional GEP publication editions, e.g. [2025] or [2022, 2026].
        These are REPORT YEARS, not historical observation years.

    force_rerank:
        None  -> use policy
        True  -> force LLM reranking
        False -> skip LLM reranking

    Returns
    -------
    Agent-safe evidence package. No final answer is generated here.
    """
    if not query or not str(query).strip():
        raise ValueError("query cannot be empty.")

    query = str(query).strip()
    years = validate_report_years(report_years or [])

    rerank_decision = should_rerank(
        report_years=years,
        force_rerank=force_rerank,
    )

    retrieval_k = (
        RERANK_CANDIDATE_K
        if rerank_decision["apply"]
        else DIRECT_RETRIEVAL_K
    )

    total_start = time.perf_counter()

    retrieval = hybrid_retrieve(
        query=query,
        report_years=years,
        num_results=retrieval_k,
    )

    candidates = retrieval["candidates"]

    rerank_latency_ms = 0.0
    rerank_calls = 0

    if rerank_decision["apply"]:
        reranked = llm_rerank(
            query=query,
            candidates=candidates,
            top_k=max(
                FINAL_TOP_K,
                MAX_EVIDENCE_ITEMS,
            ),
        )

        ranked_candidates = reranked["results"]
        rerank_latency_ms = reranked[
            "rerank_latency_ms"
        ]
        rerank_calls = reranked[
            "rerank_calls"
        ]

    else:
        ranked_candidates = candidates[
            :max(
                FINAL_TOP_K,
                MAX_EVIDENCE_ITEMS,
            )
        ]

    package = build_evidence_package(
        query=query,
        candidates=ranked_candidates,
        report_years=years,
    )

    total_latency_ms = (
        time.perf_counter() - total_start
    ) * 1000

    return {
        "tool": "research_gep",
        "status": "OK",
        "query": query,
        "report_years": years,
        "search_filters": retrieval[
            "search_filters"
        ],
        "retrieval_method": "HYBRID",
        "index_name": INDEX_NAME,
        "candidate_count": retrieval[
            "candidate_count"
        ],
        "rerank_applied": rerank_decision[
            "apply"
        ],
        "rerank_reason": rerank_decision[
            "reason"
        ],
        "rerank_calls": rerank_calls,
        "evidence": package["evidence"],
        "allowed_evidence_ids": package[
            "allowed_evidence_ids"
        ],
        "evidence_count": package[
            "evidence_count"
        ],
        "estimated_context_tokens": package[
            "estimated_context_tokens"
        ],
        "exact_duplicates_removed": package[
            "exact_duplicates_removed"
        ],
        "near_duplicates_removed": package[
            "near_duplicates_removed"
        ],
        "budget_drops": package[
            "budget_drops"
        ],
        "retrieval_latency_ms": retrieval[
            "retrieval_latency_ms"
        ],
        "rerank_latency_ms": rerank_latency_ms,
        "total_latency_ms": round(
            total_latency_ms,
            2,
        ),
    }

print("research_gep() loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Single-report production smoke test
# MAGIC
# MAGIC Expected behavior: HYBRID retrieval + 2025 filter + no reranking by default.

# COMMAND ----------

single_report_result = research_gep(
    query="What risks did the World Bank identify for South Asia in 2025?",
    report_years=[2025],
)

print("Status:", single_report_result["status"])
print("Years:", single_report_result["report_years"])
print("Rerank applied:", single_report_result["rerank_applied"])
print("Rerank reason:", single_report_result["rerank_reason"])
print("Candidates:", single_report_result["candidate_count"])
print("Evidence:", single_report_result["evidence_count"])
print("Context tokens:", single_report_result["estimated_context_tokens"])
print("Retrieval ms:", single_report_result["retrieval_latency_ms"])
print("Total ms:", single_report_result["total_latency_ms"])

assert single_report_result["status"] == "OK"
assert single_report_result["report_years"] == [2025]
assert single_report_result["rerank_applied"] is False
assert single_report_result["evidence_count"] > 0
assert single_report_result["evidence_count"] <= MAX_EVIDENCE_ITEMS

for item in single_report_result["evidence"]:
    assert int(item["report_year"]) == 2025
    assert item["evidence_id"].startswith("E")
    assert item["chunk_id"] is not None
    assert item["document_id"] is not None
    assert item["page_start"] is not None
    assert item["text"].strip()

print("Single-report research tool validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Show citation-ready single-report evidence

# COMMAND ----------

for item in single_report_result["evidence"]:
    print("=" * 100)
    print(
        f"{item['evidence_id']} | "
        f"report={item['report_year']} | "
        f"document={item['document_id']} | "
        f"pages={item['page_start']}-{item['page_end']} | "
        f"chunk={item['chunk_id']}"
    )
    print("-" * 100)
    print(item["text"][:1000])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Temporal production smoke test
# MAGIC
# MAGIC Expected behavior: report-year filter for 2022 + 2026 and conditional reranking.

# COMMAND ----------

temporal_result = research_gep(
    query=(
        "How did the World Bank's assessment of global economic risks "
        "change between 2022 and 2026?"
    ),
    report_years=[2022, 2026],
)

print("Status:", temporal_result["status"])
print("Years:", temporal_result["report_years"])
print("Rerank applied:", temporal_result["rerank_applied"])
print("Rerank reason:", temporal_result["rerank_reason"])
print("Rerank calls:", temporal_result["rerank_calls"])
print("Candidates:", temporal_result["candidate_count"])
print("Evidence:", temporal_result["evidence_count"])
print("Retrieval ms:", temporal_result["retrieval_latency_ms"])
print("Rerank ms:", temporal_result["rerank_latency_ms"])
print("Total ms:", temporal_result["total_latency_ms"])

assert temporal_result["status"] == "OK"
assert temporal_result["report_years"] == [2022, 2026]
assert temporal_result["rerank_applied"] is True
assert temporal_result["evidence_count"] > 0

returned_years = {
    int(item["report_year"])
    for item in temporal_result["evidence"]
}

assert returned_years.issubset({2022, 2026})

print("Temporal research tool validation passed.")
print("Returned evidence years:", sorted(returned_years))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Show temporal evidence

# COMMAND ----------

for item in temporal_result["evidence"]:
    print("=" * 100)
    print(
        f"{item['evidence_id']} | "
        f"report={item['report_year']} | "
        f"pages={item['page_start']}-{item['page_end']} | "
        f"retrieval_rank={item['retrieval_rank']} | "
        f"rerank_rank={item['rerank_rank']} | "
        f"rerank_score={item['rerank_score']}"
    )
    print("-" * 100)
    print(item["text"][:1000])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Agent-facing schema
# MAGIC
# MAGIC The future Supervisor/Research Agent can call this function using a constrained schema.

# COMMAND ----------

RESEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "research_gep",
        "description": (
            "Retrieve citation-ready evidence from the World Bank Global "
            "Economic Prospects January reports for qualitative outlook, "
            "risk, policy, regional, country, and temporal research questions. "
            "Use report_years for GEP publication editions, not macroeconomic "
            "observation years. This tool returns evidence and provenance; "
            "it does not generate the final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The qualitative GEP research question."
                    ),
                },
                "report_years": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "enum": [
                            2022,
                            2023,
                            2024,
                            2025,
                            2026,
                        ],
                    },
                    "uniqueItems": True,
                    "description": (
                        "Optional GEP publication editions to search."
                    ),
                },
                "force_rerank": {
                    "type": "boolean",
                    "description": (
                        "Optional override. Normally omit this so the "
                        "validated conditional reranking policy is used."
                    ),
                },
            },
            "required": [
                "query"
            ],
            "additionalProperties": False,
        },
    },
}

print(
    json.dumps(
        RESEARCH_TOOL_SCHEMA,
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Controlled research-tool registry and executor

# COMMAND ----------

RESEARCH_TOOL_REGISTRY = {
    "research_gep": {
        "function": research_gep,
        "schema": RESEARCH_TOOL_SCHEMA,
    }
}


def execute_research_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute only registered research tools.
    """
    if tool_name not in RESEARCH_TOOL_REGISTRY:
        raise ValueError(
            f"Research tool '{tool_name}' is not registered."
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be a dictionary."
        )

    return RESEARCH_TOOL_REGISTRY[
        tool_name
    ]["function"](**arguments)


executor_test = execute_research_tool(
    tool_name="research_gep",
    arguments={
        "query": (
            "What risks did the World Bank identify "
            "for South Asia in 2025?"
        ),
        "report_years": [2025],
        "force_rerank": False,
    },
)

assert executor_test["status"] == "OK"
assert executor_test["evidence_count"] > 0

print("Controlled research-tool executor passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Security test — reject unregistered research tools

# COMMAND ----------

unregistered_rejected = False

try:
    execute_research_tool(
        tool_name="search_anything",
        arguments={
            "query": "test"
        },
    )

except ValueError as exc:
    unregistered_rejected = True
    print("Expected rejection:")
    print(str(exc))

assert unregistered_rejected

print("Unregistered research-tool guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Final validation suite

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
    "ai_search_connection",
    single_year_test["candidate_count"] > 0,
    (
        f"{single_year_test['candidate_count']} "
        "single-report candidates retrieved"
    ),
)

record_test(
    "single_report_filter",
    all(
        int(item["report_year"]) == 2025
        for item in single_report_result["evidence"]
    ),
    (
        f"{single_report_result['evidence_count']} "
        "citation-ready 2025 evidence items"
    ),
)

record_test(
    "single_report_rerank_policy",
    single_report_result["rerank_applied"] is False,
    single_report_result["rerank_reason"],
)

record_test(
    "temporal_filter",
    all(
        int(item["report_year"]) in {2022, 2026}
        for item in temporal_result["evidence"]
    ),
    (
        "Temporal evidence restricted to requested "
        f"editions: {sorted(returned_years)}"
    ),
)

record_test(
    "temporal_rerank_policy",
    temporal_result["rerank_applied"] is True,
    temporal_result["rerank_reason"],
)

record_test(
    "evidence_provenance",
    all(
        item["chunk_id"] is not None
        and item["document_id"] is not None
        and item["report_year"] is not None
        and item["page_start"] is not None
        for item in (
            single_report_result["evidence"]
            + temporal_result["evidence"]
        )
    ),
    "All tested evidence has chunk/document/year/page provenance",
)

record_test(
    "context_budget",
    (
        single_report_result["estimated_context_tokens"]
        <= MAX_CONTEXT_TOKENS
        and temporal_result["estimated_context_tokens"]
        <= MAX_CONTEXT_TOKENS
    ),
    (
        f"single={single_report_result['estimated_context_tokens']}, "
        f"temporal={temporal_result['estimated_context_tokens']}, "
        f"limit={MAX_CONTEXT_TOKENS}"
    ),
)

record_test(
    "tool_allowlist",
    unregistered_rejected,
    "Unregistered research tool rejected",
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
    f"Research-tool validation failures: {failed_tests}"
)

print("All research-tool validation tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Final status

# COMMAND ----------

print("")
print("02_research_rag_tool COMPLETE")
print("")
print("Validated path:")
print(
    "agent-safe query -> report-year validation -> "
    "HYBRID AI Search -> conditional rerank -> "
    "dedupe/token budget -> citation-ready evidence"
)
print("")
print("Production research function:")
print(" - research_gep(query, report_years=None, force_rerank=None)")
print("")
print("Important:")
print(" - Reuses the locked structure-aware Qwen3 AI Search index")
print(" - Does not rebuild embeddings or vector indexes")
print(" - Uses report-year metadata filtering")
print(" - Skips expensive reranking for ordinary single-report queries")
print(" - Applies reranking to multi-report temporal comparisons")
print(" - Returns evidence, not a final synthesized answer")
print(" - Preserves document/chunk/page provenance")
print("")
print("Next notebook after this passes:")
print("05_tools_agents / 03_supervisor_agent")