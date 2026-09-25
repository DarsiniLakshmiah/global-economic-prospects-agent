# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 08_online_retrieval_pipeline
# MAGIC
# MAGIC Purpose:
# MAGIC - Connect Notebook 07 query understanding to the selected production retrieval stack.
# MAGIC - Query the Structure V1 + Qwen3 Databricks AI Search index.
# MAGIC - Use HYBRID retrieval with validated metadata filters.
# MAGIC - Apply the LLM relevance reranker conditionally, not blindly.
# MAGIC - Return a typed retrieval result that Notebook 09 can turn into grounded context.
# MAGIC
# MAGIC This notebook does NOT generate the final answer.
# MAGIC It is the online evidence-retrieval layer.

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-ai-search databricks-openai pydantic mlflow

# COMMAND ----------

# If the install above changed packages on your cluster, uncomment once and rerun the notebook.
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Imports and production configuration

# COMMAND ----------

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from pydantic import BaseModel, Field, field_validator
from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI

# Selected by our offline retrieval experiments.
AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
INDEX_NAME = "worldbank_ai.rag.gep_structure_v1_qwen3_index"

# Actual workspace endpoint already validated in prior notebooks.
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

SUPPORTED_REPORT_YEARS = [2022, 2023, 2024, 2025, 2026]

# Retrieval policy.
FINAL_TOP_K = 6
DIRECT_RETRIEVAL_K = 10
RERANK_CANDIDATE_K = 20
RERANK_BATCH_SIZE = 10
MAX_TEXT_CHARS_PER_CANDIDATE = 3500

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_online_retrieval"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

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

print("Online retrieval configuration loaded.")
print("AI Search endpoint:", AI_SEARCH_ENDPOINT)
print("Index:", INDEX_NAME)
print("LLM reranker endpoint:", LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Typed online retrieval contract
# MAGIC
# MAGIC Notebook 07 already produces a typed query-understanding object.
# MAGIC This notebook accepts either that object or an equivalent dictionary.

# COMMAND ----------

class RetrievalRequest(BaseModel):
    original_query: str
    resolved_query: str
    route_hint: str = "rag"
    report_years: List[int] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    indicators: List[str] = Field(default_factory=list)
    is_temporal_comparison: bool = False
    is_followup: bool = False
    used_llm: bool = False

    @field_validator("report_years")
    @classmethod
    def validate_report_years(cls, years: List[int]) -> List[int]:
        clean = sorted(set(int(y) for y in years))
        invalid = [y for y in clean if y not in SUPPORTED_REPORT_YEARS]
        if invalid:
            raise ValueError(f"Unsupported GEP report year(s): {invalid}")
        return clean


class RetrievedChunk(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    report_year: Optional[int] = None
    edition_status: Optional[str] = None
    chapter: Optional[str] = None
    region: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    content_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_text: str = ""
    retrieval_text: str = ""
    retrieval_rank: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: Optional[int] = None


class RetrievalResult(BaseModel):
    request: RetrievalRequest
    search_filters: Dict[str, Any] = Field(default_factory=dict)
    rerank_applied: bool
    rerank_reason: str
    candidate_count: int
    final_chunks: List[RetrievedChunk]
    retrieval_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Clients and index readiness

# COMMAND ----------

ai_search_client = AISearchClient()
index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
)

llm_client = DatabricksOpenAI()

index_description = index.describe()
index_state = (
    index_description.get("status", {}).get("detailed_state")
    or index_description.get("status", {}).get("status")
    or "UNKNOWN"
)

print("Index state:", index_state)

if not str(index_state).upper().startswith("ONLINE"):
    raise RuntimeError(f"AI Search index is not ready: {index_state}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Normalize Notebook 07 output

# COMMAND ----------

def to_plain_dict(value: Any) -> Dict[str, Any]:
    """Convert a Pydantic model, dict, or simple object into a plain dictionary."""
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Unsupported query-understanding object: {type(value)}")


def make_retrieval_request(query_understanding: Any) -> RetrievalRequest:
    """Map Notebook 07 output into the stable online retrieval contract."""
    raw = to_plain_dict(query_understanding)

    return RetrievalRequest(
        original_query=str(raw.get("original_query") or raw.get("query") or "").strip(),
        resolved_query=str(
            raw.get("resolved_query")
            or raw.get("standalone_query")
            or raw.get("original_query")
            or raw.get("query")
            or ""
        ).strip(),
        route_hint=str(raw.get("route_hint") or "rag"),
        report_years=list(raw.get("report_years") or []),
        regions=list(raw.get("regions") or []),
        sections=list(raw.get("sections") or []),
        countries=list(raw.get("countries") or []),
        indicators=list(raw.get("indicators") or []),
        is_temporal_comparison=bool(raw.get("is_temporal_comparison", False)),
        is_followup=bool(raw.get("is_followup", False)),
        used_llm=bool(raw.get("used_llm", False)),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Build safe AI Search metadata filters
# MAGIC
# MAGIC We only push filters that map directly to indexed metadata.
# MAGIC We do not invent filters from free-form text here.

# COMMAND ----------

def build_search_filters(request: RetrievalRequest) -> Dict[str, Any]:
    """Build Standard-endpoint AI Search filters from validated metadata."""
    filters: Dict[str, Any] = {}

    # One year -> exact filter. Multiple years -> IN-style list.
    if len(request.report_years) == 1:
        filters["report_year"] = int(request.report_years[0])
    elif len(request.report_years) > 1:
        filters["report_year"] = [int(y) for y in request.report_years]

    # Region metadata is useful only when Notebook 07 extracted a canonical region.
    if len(request.regions) == 1:
        filters["region"] = request.regions[0]
    elif len(request.regions) > 1:
        filters["region"] = list(request.regions)

    # Section metadata is less complete than year/region metadata.
    # Only use it when there is exactly one explicit section hint.
    if len(request.sections) == 1:
        filters["section"] = request.sections[0]

    return filters

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Parse AI Search responses robustly

# COMMAND ----------

def parse_search_response(response: Dict[str, Any]) -> List[RetrievedChunk]:
    """Convert the AI Search manifest/data_array response into typed chunks."""
    manifest_columns = response.get("manifest", {}).get("columns", [])
    column_names = [c.get("name") if isinstance(c, dict) else str(c) for c in manifest_columns]
    rows = response.get("result", {}).get("data_array", []) or []

    parsed: List[RetrievedChunk] = []

    for rank, row in enumerate(rows, start=1):
        values = list(row)

        # AI Search commonly appends a score after the requested columns.
        metadata_values = values[: len(column_names)]
        score = values[len(column_names)] if len(values) > len(column_names) else None
        record = dict(zip(column_names, metadata_values))

        parsed.append(
            RetrievedChunk(
                chunk_id=str(record.get("chunk_id")),
                parent_chunk_id=record.get("parent_chunk_id"),
                document_id=record.get("document_id"),
                report_year=int(record["report_year"]) if record.get("report_year") is not None else None,
                edition_status=record.get("edition_status"),
                chapter=record.get("chapter"),
                region=record.get("region"),
                section=record.get("section"),
                subsection=record.get("subsection"),
                content_type=record.get("content_type"),
                page_start=int(record["page_start"]) if record.get("page_start") is not None else None,
                page_end=int(record["page_end"]) if record.get("page_end") is not None else None,
                chunk_text=str(record.get("chunk_text") or ""),
                retrieval_text=str(record.get("retrieval_text") or record.get("chunk_text") or ""),
                retrieval_rank=rank,
                retrieval_score=float(score) if isinstance(score, (int, float)) else None,
            )
        )

    return parsed

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Hybrid retrieval with controlled filter fallback
# MAGIC
# MAGIC Important:
# MAGIC - Year filters were strongly validated by Notebook 05, so we never silently remove them.
# MAGIC - Region/section metadata is useful but incomplete, so if an over-specific filter returns too few
# MAGIC   candidates we retry with year-only filtering and let semantic/keyword retrieval recover evidence.

# COMMAND ----------

def run_hybrid_search(
    query: str,
    filters: Dict[str, Any],
    num_results: int,
) -> Tuple[List[RetrievedChunk], float]:
    start = time.perf_counter()

    kwargs = {
        "query_text": query,
        "columns": RETURN_COLUMNS,
        "num_results": num_results,
        "query_type": "HYBRID",
    }
    if filters:
        kwargs["filters"] = filters

    response = index.similarity_search(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return parse_search_response(response), latency_ms


def retrieve_candidates(
    request: RetrievalRequest,
    num_results: int,
) -> Tuple[List[RetrievedChunk], Dict[str, Any], float]:
    filters = build_search_filters(request)
    chunks, latency_ms = run_hybrid_search(request.resolved_query, filters, num_results)

    # Controlled fallback for sparse/inexact structural metadata.
    # Preserve report_year because our offline evaluation showed it materially improves retrieval.
    has_soft_filters = "region" in filters or "section" in filters
    minimum_useful_candidates = min(FINAL_TOP_K, num_results)

    if has_soft_filters and len(chunks) < minimum_useful_candidates:
        year_only_filters = {
            k: v for k, v in filters.items() if k == "report_year"
        }
        retry_chunks, retry_latency_ms = run_hybrid_search(
            request.resolved_query,
            year_only_filters,
            num_results,
        )
        chunks = retry_chunks
        filters = year_only_filters
        latency_ms += retry_latency_ms

    return chunks, filters, latency_ms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Conditional reranking policy
# MAGIC
# MAGIC Notebook 06 showed that LLM reranking can improve ranking quality but adds meaningful latency.
# MAGIC We therefore rerank only when the query is complex enough to justify it.

# COMMAND ----------

def should_rerank(request: RetrievalRequest) -> Tuple[bool, str]:
    """Return whether to invoke the LLM relevance reranker and a traceable reason."""
    route = request.route_hint.lower()

    if request.is_temporal_comparison or route == "temporal_rag":
        return True, "temporal comparison benefits from cross-edition relevance ordering"

    if route == "hybrid":
        return True, "hybrid question contains multiple evidence needs"

    if len(request.report_years) > 1:
        return True, "multiple report editions requested"

    if len(request.regions) > 1:
        return True, "multiple regions requested"

    if len(request.sections) > 1:
        return True, "multiple document sections requested"

    if request.is_followup and request.used_llm:
        return True, "ambiguous conversational follow-up was resolved with the LLM"

    return False, "single-scope query uses validated hybrid retrieval directly"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. LLM relevance reranker
# MAGIC
# MAGIC This reuses the same relevance-only idea from Notebook 06.
# MAGIC The model does not answer the question. It only scores retrieved evidence.

# COMMAND ----------

RERANK_SYSTEM_PROMPT = """
You are a relevance reranker for World Bank Global Economic Prospects evidence.

Given one user query and a list of retrieved chunks, score every chunk from 0 to 4:
0 = irrelevant
1 = weakly related
2 = partially useful
3 = relevant
4 = highly relevant or directly useful

Rules:
- Judge relevance to the user's information need, not writing quality.
- Use the supplied report year, region, section, pages, and text.
- Do not answer the user's question.
- Do not rewrite the chunks.
- Score every supplied chunk exactly once.
- Return JSON only in this form:
  {"scores": [{"chunk_id": "...", "score": 0}]}
""".strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in reranker output: {cleaned[:500]}")

    return json.loads(cleaned[start : end + 1])


def rerank_batch(query: str, chunks: List[RetrievedChunk]) -> Dict[str, float]:
    candidate_payload = []
    for chunk in chunks:
        candidate_payload.append(
            {
                "chunk_id": chunk.chunk_id,
                "report_year": chunk.report_year,
                "region": chunk.region,
                "section": chunk.section,
                "pages": [chunk.page_start, chunk.page_end],
                "text": chunk.retrieval_text[:MAX_TEXT_CHARS_PER_CANDIDATE],
            }
        )

    user_payload = {
        "query": query,
        "candidates": candidate_payload,
    }

    response = llm_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=1200,
    )

    raw_text = response.choices[0].message.content
    parsed = parse_json_object(raw_text)
    scores = parsed.get("scores")

    if not isinstance(scores, list):
        raise ValueError("Reranker output is missing a 'scores' list.")

    expected_ids = [c.chunk_id for c in chunks]
    returned: Dict[str, float] = {}

    for item in scores:
        chunk_id = str(item.get("chunk_id", ""))
        score = float(item.get("score"))

        if chunk_id not in expected_ids:
            raise ValueError(f"Reranker returned unexpected chunk_id: {chunk_id}")
        if chunk_id in returned:
            raise ValueError(f"Reranker returned duplicate chunk_id: {chunk_id}")
        if score < 0 or score > 4:
            raise ValueError(f"Reranker score outside 0-4 range: {score}")

        returned[chunk_id] = score

    missing = [chunk_id for chunk_id in expected_ids if chunk_id not in returned]
    if missing:
        raise ValueError(f"Reranker did not score all candidates. Missing: {missing}")

    return returned


def rerank_candidates(
    query: str,
    chunks: List[RetrievedChunk],
) -> Tuple[List[RetrievedChunk], float]:
    start = time.perf_counter()
    score_map: Dict[str, float] = {}

    for batch_start in range(0, len(chunks), RERANK_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + RERANK_BATCH_SIZE]
        score_map.update(rerank_batch(query, batch))

    reranked = []
    for chunk in chunks:
        updated = chunk.model_copy(deep=True)
        updated.rerank_score = score_map[chunk.chunk_id]
        reranked.append(updated)

    # Primary key = LLM relevance score; tie-breaker = original retrieval rank.
    reranked.sort(key=lambda c: (-float(c.rerank_score), c.retrieval_rank))

    latency_ms = (time.perf_counter() - start) * 1000.0
    return reranked, latency_ms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Main online retrieval function

# COMMAND ----------

def online_retrieve(query_understanding: Any) -> RetrievalResult:
    total_start = time.perf_counter()
    request = make_retrieval_request(query_understanding)

    if not request.resolved_query:
        raise ValueError("resolved_query is empty; Notebook 07 must provide a standalone query.")

    rerank_applied, rerank_reason = should_rerank(request)
    candidate_k = RERANK_CANDIDATE_K if rerank_applied else DIRECT_RETRIEVAL_K

    candidates, search_filters, retrieval_latency_ms = retrieve_candidates(
        request=request,
        num_results=candidate_k,
    )

    if not candidates:
        return RetrievalResult(
            request=request,
            search_filters=search_filters,
            rerank_applied=False,
            rerank_reason="no retrieval candidates returned",
            candidate_count=0,
            final_chunks=[],
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            rerank_latency_ms=0.0,
            total_latency_ms=round((time.perf_counter() - total_start) * 1000.0, 2),
        )

    rerank_latency_ms = 0.0

    if rerank_applied:
        ranked, rerank_latency_ms = rerank_candidates(request.resolved_query, candidates)
    else:
        ranked = candidates

    final_chunks = ranked[:FINAL_TOP_K]
    for final_rank, chunk in enumerate(final_chunks, start=1):
        chunk.final_rank = final_rank

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    return RetrievalResult(
        request=request,
        search_filters=search_filters,
        rerank_applied=rerank_applied,
        rerank_reason=rerank_reason,
        candidate_count=len(candidates),
        final_chunks=final_chunks,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        rerank_latency_ms=round(rerank_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Smoke tests
# MAGIC
# MAGIC These tests are self-contained so Notebook 08 can be validated even before Notebook 07 is imported as a Python module.

# COMMAND ----------

SMOKE_TESTS = [
    {
        "name": "single_year_regional_rag",
        "query_understanding": {
            "original_query": "What risks did the World Bank identify for South Asia in January 2025?",
            "resolved_query": "What risks did the World Bank identify for South Asia in January 2025?",
            "route_hint": "rag",
            "report_years": [2025],
            "regions": ["South Asia"],
            "sections": ["Risks"],
            "countries": [],
            "indicators": [],
            "is_temporal_comparison": False,
            "is_followup": False,
            "used_llm": False,
        },
    },
    {
        "name": "temporal_rag",
        "query_understanding": {
            "original_query": "How did the World Bank's assessment of global economic risks change between 2022 and 2026?",
            "resolved_query": "How did the World Bank's assessment of global economic risks change between 2022 and 2026?",
            "route_hint": "temporal_rag",
            "report_years": [2022, 2026],
            "regions": [],
            "sections": ["Risks"],
            "countries": [],
            "indicators": [],
            "is_temporal_comparison": True,
            "is_followup": False,
            "used_llm": False,
        },
    },
]

for test in SMOKE_TESTS:
    print("\n" + "=" * 100)
    print("TEST:", test["name"])
    result = online_retrieve(test["query_understanding"])

    print("resolved_query:", result.request.resolved_query)
    print("filters:", result.search_filters)
    print("rerank_applied:", result.rerank_applied)
    print("rerank_reason:", result.rerank_reason)
    print("candidate_count:", result.candidate_count)
    print("retrieval_latency_ms:", result.retrieval_latency_ms)
    print("rerank_latency_ms:", result.rerank_latency_ms)
    print("total_latency_ms:", result.total_latency_ms)

    for chunk in result.final_chunks:
        print(
            f"  final_rank={chunk.final_rank} "
            f"retrieval_rank={chunk.retrieval_rank} "
            f"rerank_score={chunk.rerank_score} "
            f"year={chunk.report_year} "
            f"region={chunk.region} "
            f"section={chunk.section} "
            f"pages={chunk.page_start}-{chunk.page_end} "
            f"chunk_id={chunk.chunk_id}"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validation assertions

# COMMAND ----------

# Validate the simple query path.
simple_result = online_retrieve(SMOKE_TESTS[0]["query_understanding"])
assert simple_result.final_chunks, "Simple retrieval returned no evidence."
assert simple_result.search_filters.get("report_year") == 2025, "Expected 2025 year filter."
assert all(c.report_year == 2025 for c in simple_result.final_chunks), "Year filter was not respected."

# Validate the temporal path.
temporal_result = online_retrieve(SMOKE_TESTS[1]["query_understanding"])
assert temporal_result.final_chunks, "Temporal retrieval returned no evidence."
assert temporal_result.rerank_applied is True, "Temporal query should use reranking."
assert set(c.report_year for c in temporal_result.final_chunks).issubset({2022, 2026}), \
    "Temporal retrieval returned an unexpected report year."

print("Online retrieval validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. MLflow traceable smoke run
# MAGIC
# MAGIC We log online retrieval configuration and latency separately from the offline benchmark.

# COMMAND ----------

with mlflow.start_run(run_name="online_retrieval_smoke"):
    mlflow.log_params(
        {
            "index_name": INDEX_NAME,
            "query_type": "HYBRID",
            "final_top_k": FINAL_TOP_K,
            "direct_retrieval_k": DIRECT_RETRIEVAL_K,
            "rerank_candidate_k": RERANK_CANDIDATE_K,
            "rerank_model": LLM_ENDPOINT,
            "rerank_policy": "conditional",
        }
    )

    mlflow.log_metrics(
        {
            "simple_retrieval_latency_ms": simple_result.retrieval_latency_ms,
            "simple_total_latency_ms": simple_result.total_latency_ms,
            "temporal_retrieval_latency_ms": temporal_result.retrieval_latency_ms,
            "temporal_rerank_latency_ms": temporal_result.rerank_latency_ms,
            "temporal_total_latency_ms": temporal_result.total_latency_ms,
        }
    )

print("MLflow online retrieval smoke run logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Example handoff to Notebook 09
# MAGIC
# MAGIC Notebook 09 should consume `RetrievalResult.final_chunks` and build a bounded evidence context.
# MAGIC It should preserve chunk/document/page provenance for citations.

# COMMAND ----------

example_handoff = temporal_result.model_dump()

print("Example output keys:", list(example_handoff.keys()))
print("Final evidence chunks:", len(example_handoff["final_chunks"]))
print("\n08_online_retrieval_pipeline COMPLETE")