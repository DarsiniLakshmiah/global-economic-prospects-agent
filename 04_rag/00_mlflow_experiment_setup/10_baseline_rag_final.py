# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 10_baseline_rag
# MAGIC
# MAGIC Production-style baseline RAG generation for the Global Economic Prospects Intelligence Agent.
# MAGIC
# MAGIC This notebook is intentionally self-contained:
# MAGIC
# MAGIC `query -> AI Search HYBRID retrieval -> provenance validation -> context builder -> LLM -> citation validation -> MLflow`
# MAGIC
# MAGIC It does not depend on `%run`, notebook session state, widgets, copied JSON, or an intermediate Delta handoff.
# MAGIC
# MAGIC Notebook 08 and Notebook 09 remain locked experiments. The context-builder functions below preserve the validated Notebook 09 contract.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Imports and configuration
# MAGIC
# MAGIC These packages were already used successfully in the earlier notebooks.
# MAGIC Do not upgrade/reinstall packages here unless your cluster actually reports a missing import.

# COMMAND ----------

# Install the two Databricks SDK packages used by this notebook.
# We are NOT upgrading mlflow/pydantic or changing the whole environment.

%pip install -q databricks-ai-search databricks-openai

# COMMAND ----------

# Restart Python so the newly installed packages are loaded cleanly.
dbutils.library.restartPython()

# COMMAND ----------

# Verify that this session can reach our existing AI Search index.

from databricks.ai_search.client import AISearchClient

ai_search_client = AISearchClient()

search_index = ai_search_client.get_index(
    endpoint_name="worldbank-gep-ai-search",
    index_name="worldbank_ai.rag.gep_structure_v1_qwen3_index",
)

print("AI Search connection successful")

# COMMAND ----------

# Verify Databricks model-serving authentication.

from databricks_openai import DatabricksOpenAI

llm_client = DatabricksOpenAI()

response = llm_client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: MODEL_OK"
        }
    ],
    temperature=0.0,
    max_tokens=20,
)

print(response.choices[0].message.content)

# COMMAND ----------

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from pydantic import BaseModel, Field, field_validator

from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_baseline_rag"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
AI_SEARCH_INDEX = "worldbank_ai.rag.gep_structure_v1_qwen3_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Same validated context-builder settings as Notebook 09.
MAX_EVIDENCE_ITEMS = 6
MAX_CONTEXT_TOKENS = 6000
CHARS_PER_TOKEN = 4.0
MAX_TOKENS_PER_EVIDENCE = 1400
NEAR_DUPLICATE_JACCARD = 0.82

# Baseline retrieval configuration selected by the earlier retrieval evaluation.
RETRIEVAL_TOP_K = 6
QUERY_TYPE = "HYBRID"

# Generation configuration.
MAX_ANSWER_TOKENS = 1200

print("Notebook 10 configuration loaded.")
print("AI Search endpoint:", AI_SEARCH_ENDPOINT)
print("AI Search index:", AI_SEARCH_INDEX)
print("LLM endpoint:", LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Test request
# MAGIC
# MAGIC Start with the same real 2025 South Asia risk question used to validate Notebook 09.
# MAGIC Change these two values later to test another GEP edition/question.

# COMMAND ----------

USER_QUERY = "What risks did the World Bank identify for South Asia in 2025?"
REPORT_YEAR = 2025

if REPORT_YEAR not in {2022, 2023, 2024, 2025, 2026}:
    raise ValueError("REPORT_YEAR must be one of 2022, 2023, 2024, 2025, 2026.")

print("Query:", USER_QUERY)
print("Report year filter:", REPORT_YEAR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Context data models
# MAGIC
# MAGIC These preserve Notebook 09's citation-ready evidence contract.

# COMMAND ----------

class EvidenceItem(BaseModel):
    evidence_id: str

    chunk_id: str
    parent_chunk_id: Optional[str] = None
    document_id: str

    report_year: int
    edition_status: Optional[str] = None

    chapter: Optional[str] = None
    region: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    content_type: Optional[str] = None

    page_start: int
    page_end: int

    text: str

    retrieval_rank: Optional[int] = None
    rerank_score: Optional[float] = None

    estimated_tokens: int

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value):
        value = (value or "").strip()
        if not value:
            raise ValueError("Evidence text cannot be empty.")
        return value


class ContextPackage(BaseModel):
    original_query: str
    resolved_query: str

    route_hint: str = "rag"
    report_years: List[int] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)

    evidence: List[EvidenceItem] = Field(default_factory=list)

    context_text: str
    estimated_context_tokens: int

    input_evidence_count: int
    final_evidence_count: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    budget_drops: int


class CitationRecord(BaseModel):
    evidence_id: str
    document_id: str
    report_year: int
    edition_status: Optional[str] = None
    page_start: int
    page_end: int
    chunk_id: str


class BaselineRAGResult(BaseModel):
    query: str
    answer: str
    cited_evidence_ids: List[str]
    citations: List[CitationRecord]

    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    retrieved_evidence_count: int
    final_evidence_count: int
    estimated_context_tokens: int

    citation_validation_passed: bool
    answer_has_citations: bool


print("Data models loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Context-builder helpers
# MAGIC
# MAGIC This is the validated Notebook 09 behavior:
# MAGIC source text is normalized/truncated for budget only, exact duplicates are removed,
# MAGIC and near-duplicate suppression requires matching document/year plus overlapping provenance.

# COMMAND ----------

def to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    raise TypeError(
        f"Expected dict or Pydantic-like object, got {type(value)}"
    )


def get_first(
    row: Dict[str, Any],
    names: List[str],
    default=None,
):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]

    return default


def normalize_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    # Deterministic estimate for context budgeting only.
    text = text or ""

    if not text:
        return 0

    return max(
        1,
        int(math.ceil(len(text) / CHARS_PER_TOKEN)),
    )


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
) -> str:
    # Truncate without rewriting source evidence.
    text = normalize_whitespace(text)

    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    truncated = text[:max_chars]

    last_space = truncated.rfind(" ")

    if last_space >= int(max_chars * 0.85):
        truncated = truncated[:last_space]

    return truncated.rstrip()


def normalized_fingerprint(text: str) -> str:
    text = normalize_whitespace(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lexical_token_set(text: str) -> set:
    normalized = normalized_fingerprint(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
    }


def jaccard_similarity(
    left: str,
    right: str,
) -> float:
    left_tokens = lexical_token_set(left)
    right_tokens = lexical_token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    union = left_tokens | right_tokens

    if not union:
        return 0.0

    return len(left_tokens & right_tokens) / len(union)


def pages_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def is_near_duplicate(
    candidate: Dict[str, Any],
    selected: Dict[str, Any],
) -> bool:
    if str(candidate["document_id"]) != str(selected["document_id"]):
        return False

    if int(candidate["report_year"]) != int(selected["report_year"]):
        return False

    same_parent = (
        candidate.get("parent_chunk_id")
        and selected.get("parent_chunk_id")
        and str(candidate.get("parent_chunk_id"))
        == str(selected.get("parent_chunk_id"))
    )

    overlapping_pages = pages_overlap(
        int(candidate["page_start"]),
        int(candidate["page_end"]),
        int(selected["page_start"]),
        int(selected["page_end"]),
    )

    if not (same_parent or overlapping_pages):
        return False

    similarity = jaccard_similarity(
        candidate["text"],
        selected["text"],
    )

    return similarity >= NEAR_DUPLICATE_JACCARD

# COMMAND ----------

def canonicalize_evidence(
    raw_evidence: List[Any],
) -> List[Dict[str, Any]]:
    canonical = []

    for position, item in enumerate(raw_evidence, start=1):
        row = to_dict(item)

        text = get_first(
            row,
            ["chunk_text", "text", "retrieval_text"],
            "",
        )
        text = normalize_whitespace(str(text))

        if not text:
            continue

        chunk_id = str(
            get_first(
                row,
                ["chunk_id"],
                f"missing_chunk_id_{position}",
            )
        )

        document_id = get_first(row, ["document_id"])
        report_year = get_first(row, ["report_year"])
        page_start = get_first(row, ["page_start"])
        page_end = get_first(row, ["page_end"], page_start)

        # Citation provenance is mandatory.
        if document_id is None:
            raise ValueError(
                f"Evidence {chunk_id} is missing document_id."
            )

        if report_year is None:
            raise ValueError(
                f"Evidence {chunk_id} is missing report_year."
            )

        if page_start is None:
            raise ValueError(
                f"Evidence {chunk_id} is missing page_start."
            )

        if page_end is None:
            page_end = page_start

        retrieval_rank = get_first(
            row,
            ["retrieval_rank", "_retrieval_rank", "rank"],
            position,
        )

        rerank_score = get_first(
            row,
            ["rerank_score", "_rerank_score"],
        )

        canonical.append({
            "chunk_id": chunk_id,
            "parent_chunk_id": get_first(row, ["parent_chunk_id"]),
            "document_id": str(document_id),
            "report_year": int(report_year),
            "edition_status": get_first(row, ["edition_status"]),
            "chapter": get_first(row, ["chapter"]),
            "region": get_first(row, ["region"]),
            "section": get_first(row, ["section"]),
            "subsection": get_first(row, ["subsection"]),
            "content_type": get_first(row, ["content_type"]),
            "page_start": int(page_start),
            "page_end": int(page_end),
            "text": text,
            "retrieval_rank": (
                int(retrieval_rank)
                if retrieval_rank is not None
                else position
            ),
            "rerank_score": (
                float(rerank_score)
                if rerank_score is not None
                else None
            ),
        })

    return canonical


def deduplicate_evidence(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    selected = []
    exact_seen = set()

    exact_removed = 0
    near_removed = 0

    for row in rows:
        fingerprint = (
            row["document_id"],
            row["report_year"],
            normalized_fingerprint(row["text"]),
        )

        if fingerprint in exact_seen:
            exact_removed += 1
            continue

        if any(
            is_near_duplicate(row, prior)
            for prior in selected
        ):
            near_removed += 1
            continue

        exact_seen.add(fingerprint)
        selected.append(row)

    return selected, exact_removed, near_removed


def select_under_budget(
    rows: List[Dict[str, Any]],
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> Tuple[List[Dict[str, Any]], int]:
    selected = []
    used_tokens = 0
    budget_drops = 0

    for row in rows:
        if len(selected) >= max_items:
            budget_drops += 1
            continue

        remaining = max_context_tokens - used_tokens

        if remaining <= 0:
            budget_drops += 1
            continue

        per_item_budget = min(
            MAX_TOKENS_PER_EVIDENCE,
            remaining,
        )

        if per_item_budget < 80:
            budget_drops += 1
            continue

        final_text = truncate_to_token_budget(
            row["text"],
            per_item_budget,
        )

        token_count = estimate_tokens(final_text)

        if not final_text or token_count <= 0:
            budget_drops += 1
            continue

        enriched = dict(row)
        enriched["text"] = final_text
        enriched["estimated_tokens"] = token_count

        selected.append(enriched)
        used_tokens += token_count

    return selected, budget_drops

# COMMAND ----------

def format_pages(
    page_start: int,
    page_end: int,
) -> str:
    if page_start == page_end:
        return f"p. {page_start}"

    return f"pp. {page_start}-{page_end}"


def format_evidence_block(
    evidence_id: str,
    row: Dict[str, Any],
) -> str:
    metadata_parts = [
        f"report_year={row['report_year']}",
        f"pages={format_pages(row['page_start'], row['page_end'])}",
        f"document_id={row['document_id']}",
        f"chunk_id={row['chunk_id']}",
    ]

    optional_fields = [
        ("edition_status", row.get("edition_status")),
        ("chapter", row.get("chapter")),
        ("region", row.get("region")),
        ("section", row.get("section")),
        ("subsection", row.get("subsection")),
    ]

    for name, value in optional_fields:
        if value is not None and str(value).strip():
            metadata_parts.append(
                f"{name}={str(value).strip()}"
            )

    metadata_line = " | ".join(metadata_parts)

    return (
        f"[{evidence_id}]\n"
        f"{metadata_line}\n"
        f"{row['text']}"
    )


def build_context_text(
    evidence_rows: List[Dict[str, Any]],
) -> str:
    blocks = []

    for i, row in enumerate(evidence_rows, start=1):
        blocks.append(
            format_evidence_block(
                evidence_id=f"E{i}",
                row=row,
            )
        )

    return "\n\n---\n\n".join(blocks)


def build_context_package(
    retrieval_result: Any,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> ContextPackage:
    result = to_dict(retrieval_result)

    # Notebook 10 uses the explicit `evidence` field.
    raw_evidence = get_first(
        result,
        ["evidence", "final_evidence", "results"],
        [],
    )

    if raw_evidence is None:
        raw_evidence = []

    if not isinstance(raw_evidence, list):
        raise TypeError("Retrieval evidence must be a list.")

    if not raw_evidence:
        raise ValueError(
            "Cannot build grounded context from zero evidence chunks."
        )

    canonical = canonicalize_evidence(raw_evidence)

    if not canonical:
        raise ValueError(
            "No valid evidence remained after canonicalization."
        )

    deduped, exact_removed, near_removed = deduplicate_evidence(
        canonical
    )

    selected, budget_drops = select_under_budget(
        rows=deduped,
        max_context_tokens=max_context_tokens,
        max_items=max_items,
    )

    if not selected:
        raise ValueError(
            "No evidence fit inside the context budget."
        )

    evidence_models = [
        EvidenceItem(
            evidence_id=f"E{i}",
            **row,
        )
        for i, row in enumerate(selected, start=1)
    ]

    context_text = build_context_text(selected)
    estimated_context_tokens = estimate_tokens(context_text)

    package = ContextPackage(
        original_query=str(
            get_first(result, ["original_query", "query"], "")
        ),
        resolved_query=str(
            get_first(
                result,
                ["resolved_query", "search_query", "query"],
                "",
            )
        ),
        route_hint=str(
            get_first(result, ["route_hint"], "rag")
        ),
        report_years=[
            int(x)
            for x in get_first(result, ["report_years"], [])
        ],
        regions=[
            str(x)
            for x in get_first(result, ["regions"], [])
        ],
        sections=[
            str(x)
            for x in get_first(result, ["sections"], [])
        ],
        evidence=evidence_models,
        context_text=context_text,
        estimated_context_tokens=estimated_context_tokens,
        input_evidence_count=len(canonical),
        final_evidence_count=len(evidence_models),
        exact_duplicates_removed=exact_removed,
        near_duplicates_removed=near_removed,
        budget_drops=budget_drops,
    )

    # Same validation envelope as Notebook 09.
    assert 0 < package.final_evidence_count <= max_items
    assert package.estimated_context_tokens <= (
        max_context_tokens + 800
    )

    return package

print("Validated context-builder functions loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Connect to the production-candidate AI Search index

# COMMAND ----------

RETRIEVAL_COLUMNS = [
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
]

ai_search_client = AISearchClient()

search_index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=AI_SEARCH_INDEX,
)

index_description = search_index.describe()

print("Connected to AI Search.")
print("Endpoint:", AI_SEARCH_ENDPOINT)
print("Index:", AI_SEARCH_INDEX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Retrieve real evidence
# MAGIC
# MAGIC We use HYBRID retrieval plus the report-year filter because the earlier retrieval evaluation selected this as the production candidate.

# COMMAND ----------

retrieval_started = time.perf_counter()

search_response = search_index.similarity_search(
    query_text=USER_QUERY,
    columns=RETRIEVAL_COLUMNS,
    num_results=RETRIEVAL_TOP_K,
    query_type=QUERY_TYPE,
    filters={
        "report_year": REPORT_YEAR,
    },
)

retrieval_latency_ms = (
    time.perf_counter() - retrieval_started
) * 1000.0

print(
    "AI Search completed in "
    f"{retrieval_latency_ms:.1f} ms."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Normalize and validate AI Search response

# COMMAND ----------

def normalize_ai_search_response(
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if response is None:
        raise ValueError("AI Search returned None.")

    if not isinstance(response, dict):
        raise TypeError(
            "Expected AI Search response to be a dictionary, "
            f"got {type(response)}"
        )

    manifest = response.get("manifest") or {}
    result = response.get("result") or {}

    columns = manifest.get("columns") or []
    data_array = result.get("data_array") or []

    if not columns:
        raise ValueError(
            "AI Search response is missing manifest columns."
        )

    if not data_array:
        raise ValueError(
            "AI Search returned zero evidence rows."
        )

    column_names = []

    for col in columns:
        if isinstance(col, dict):
            name = col.get("name")
        else:
            name = str(col)

        if not name:
            raise ValueError(
                f"Invalid AI Search manifest column: {col}"
            )

        column_names.append(name)

    normalized_rows = []

    for rank, values in enumerate(data_array, start=1):
        if len(values) < len(column_names):
            raise ValueError(
                "AI Search row has fewer values than manifest columns."
            )

        row = dict(
            zip(
                column_names,
                values[:len(column_names)],
            )
        )
        row["_retrieval_rank"] = rank
        normalized_rows.append(row)

    return normalized_rows


retrieved_rows = normalize_ai_search_response(
    search_response
)

required_fields = [
    "chunk_id",
    "document_id",
    "report_year",
    "page_start",
    "page_end",
    "chunk_text",
]

for rank, row in enumerate(retrieved_rows, start=1):
    missing = [
        field
        for field in required_fields
        if row.get(field) is None
    ]

    if missing:
        raise ValueError(
            f"Retrieved evidence rank {rank} is missing: {missing}"
        )

    if int(row["report_year"]) != REPORT_YEAR:
        raise AssertionError(
            "Year-filtered AI Search returned unexpected "
            f"report year at rank {rank}: {row['report_year']}"
        )

    if not str(row["chunk_text"]).strip():
        raise ValueError(
            f"Retrieved evidence rank {rank} has empty chunk_text."
        )

print("Retrieved chunks:", len(retrieved_rows))
print("Retrieval provenance validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Build citation-ready context
# MAGIC
# MAGIC This is the same contract validated in Notebook 09.

# COMMAND ----------

retrieval_result = {
    "original_query": USER_QUERY,
    "resolved_query": USER_QUERY,
    "route_hint": "rag",
    "report_years": [REPORT_YEAR],
    "regions": [],
    "sections": [],
    "evidence": retrieved_rows,
}

context_package = build_context_package(
    retrieval_result
)

print("Context package built.")
print("Input evidence:", context_package.input_evidence_count)
print("Final evidence:", context_package.final_evidence_count)
print(
    "Exact duplicates removed:",
    context_package.exact_duplicates_removed,
)
print(
    "Near duplicates removed:",
    context_package.near_duplicates_removed,
)
print("Budget drops:", context_package.budget_drops)
print(
    "Estimated context tokens:",
    context_package.estimated_context_tokens,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Validate context provenance before generation

# COMMAND ----------

allowed_evidence_ids = [
    item.evidence_id
    for item in context_package.evidence
]

assert allowed_evidence_ids
assert len(allowed_evidence_ids) == len(
    set(allowed_evidence_ids)
)

for item in context_package.evidence:
    assert item.document_id
    assert item.chunk_id
    assert item.report_year == REPORT_YEAR
    assert item.page_start >= 1
    assert item.page_end >= item.page_start
    assert item.text.strip()
    assert (
        f"[{item.evidence_id}]"
        in context_package.context_text
    )

print("Allowed evidence IDs:", allowed_evidence_ids)
print("Context provenance validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Grounded generation prompt
# MAGIC
# MAGIC The model receives only the user question and retrieved evidence.
# MAGIC It must use `[E#]` citations that map to the context package.

# COMMAND ----------

SYSTEM_PROMPT = """
You are a grounded research assistant for World Bank Global Economic Prospects reports.

Rules:
1. Answer the user's question using only the evidence in the supplied context.
2. Do not add facts from memory or outside knowledge.
3. Cite factual claims using the evidence IDs exactly as [E1], [E2], etc.
4. Never invent an evidence ID.
5. If the evidence is insufficient to answer an important part of the question, say so explicitly.
6. Synthesize across evidence when useful instead of copying long passages.
7. Keep the answer concise but substantive.
8. Do not include a separate references section; citations should appear next to the claims they support.
""".strip()


def build_user_prompt(
    query: str,
    context_text: str,
) -> str:
    return f"""
QUESTION
{query}

RETRIEVED EVIDENCE
{context_text}

Write a grounded answer to the QUESTION using only the RETRIEVED EVIDENCE.
""".strip()


user_prompt = build_user_prompt(
    context_package.resolved_query,
    context_package.context_text,
)

print("Generation prompt prepared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Generate the baseline RAG answer

# COMMAND ----------

llm_client = DatabricksOpenAI()

generation_started = time.perf_counter()

response = llm_client.chat.completions.create(
    model=LLM_ENDPOINT,
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ],
    temperature=0.0,
    max_tokens=MAX_ANSWER_TOKENS,
)

generation_latency_ms = (
    time.perf_counter() - generation_started
) * 1000.0

answer = (
    response.choices[0].message.content or ""
).strip()

if not answer:
    raise ValueError(
        "The LLM returned an empty answer."
    )

print("BASELINE RAG ANSWER")
print("-------------------")
print(answer)
print("")
print(
    "Generation latency:",
    f"{generation_latency_ms:.1f} ms",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Citation validation
# MAGIC
# MAGIC This validator checks citation integrity deterministically:
# MAGIC every generated `[E#]` must exist in the context package.

# COMMAND ----------

def extract_cited_evidence_ids(
    answer_text: str,
) -> List[str]:
    matches = re.findall(
        r"\[(E\d+)\]",
        answer_text or "",
    )

    # Preserve first appearance order.
    return list(dict.fromkeys(matches))


cited_evidence_ids = extract_cited_evidence_ids(
    answer
)

allowed_set = set(allowed_evidence_ids)
cited_set = set(cited_evidence_ids)

unsupported_citations = sorted(
    cited_set - allowed_set
)

answer_has_citations = len(
    cited_evidence_ids
) > 0

citation_validation_passed = (
    answer_has_citations
    and not unsupported_citations
)

if unsupported_citations:
    raise AssertionError(
        "The model generated unsupported evidence IDs: "
        f"{unsupported_citations}"
    )

if not answer_has_citations:
    raise AssertionError(
        "The answer contains no [E#] citations."
    )

print("Cited evidence IDs:", cited_evidence_ids)
print("Unsupported citations:", unsupported_citations)
print("Citation validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Resolve generated citations to document provenance

# COMMAND ----------

metadata_by_id = {
    item.evidence_id: item
    for item in context_package.evidence
}

citation_records = []

for evidence_id in cited_evidence_ids:
    item = metadata_by_id[evidence_id]

    citation_records.append(
        CitationRecord(
            evidence_id=evidence_id,
            document_id=item.document_id,
            report_year=item.report_year,
            edition_status=item.edition_status,
            page_start=item.page_start,
            page_end=item.page_end,
            chunk_id=item.chunk_id,
        )
    )

print("CITATION PROVENANCE")
print("-------------------")

for citation in citation_records:
    print(
        f"[{citation.evidence_id}] "
        f"{citation.document_id} | "
        f"{citation.report_year} | "
        f"{format_pages(citation.page_start, citation.page_end)} | "
        f"chunk={citation.chunk_id}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Build final typed result

# COMMAND ----------

total_latency_ms = (
    retrieval_latency_ms
    + generation_latency_ms
)

rag_result = BaselineRAGResult(
    query=context_package.resolved_query,
    answer=answer,
    cited_evidence_ids=cited_evidence_ids,
    citations=citation_records,
    retrieval_latency_ms=round(
        retrieval_latency_ms,
        2,
    ),
    generation_latency_ms=round(
        generation_latency_ms,
        2,
    ),
    total_latency_ms=round(
        total_latency_ms,
        2,
    ),
    retrieved_evidence_count=len(retrieved_rows),
    final_evidence_count=(
        context_package.final_evidence_count
    ),
    estimated_context_tokens=(
        context_package.estimated_context_tokens
    ),
    citation_validation_passed=(
        citation_validation_passed
    ),
    answer_has_citations=answer_has_citations,
)

print(
    rag_result.model_dump_json(
        indent=2
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. MLflow trace / experiment logging
# MAGIC
# MAGIC Log configuration, latency, citation integrity, and the actual grounded output.
# MAGIC This gives Notebook 11 a reproducible baseline to evaluate.

# COMMAND ----------

with mlflow.start_run(
    run_name="baseline_rag_real_ai_search_v1"
):
    mlflow.log_params({
        "ai_search_endpoint": AI_SEARCH_ENDPOINT,
        "ai_search_index": AI_SEARCH_INDEX,
        "query_type": QUERY_TYPE,
        "report_year_filter": REPORT_YEAR,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "llm_endpoint": LLM_ENDPOINT,
        "temperature": 0.0,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
        "context_builder_version": "notebook09_validated_v1",
    })

    mlflow.log_metrics({
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
        "retrieved_evidence_count": len(retrieved_rows),
        "final_evidence_count": (
            context_package.final_evidence_count
        ),
        "estimated_context_tokens": (
            context_package.estimated_context_tokens
        ),
        "citation_count": len(cited_evidence_ids),
        "citation_validation_passed": float(
            citation_validation_passed
        ),
    })

    # Log small text artifacts for reproducibility.
    mlflow.log_text(
        context_package.resolved_query,
        "query.txt",
    )

    mlflow.log_text(
        answer,
        "answer.txt",
    )

    mlflow.log_text(
        json.dumps(
            [
                record.model_dump()
                for record in citation_records
            ],
            indent=2,
        ),
        "citation_provenance.json",
    )

print("MLflow baseline RAG run logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Notebook 11 evaluation record
# MAGIC
# MAGIC Keep a compact in-memory contract for the next evaluation notebook.
# MAGIC No fabricated quality score is assigned here; Notebook 11 will evaluate the answer.

# COMMAND ----------

evaluation_record = {
    "query": rag_result.query,
    "answer": rag_result.answer,
    "context": context_package.context_text,
    "allowed_evidence_ids": allowed_evidence_ids,
    "cited_evidence_ids": rag_result.cited_evidence_ids,
    "citation_provenance": [
        record.model_dump()
        for record in rag_result.citations
    ],
    "report_year": REPORT_YEAR,
    "retrieval_latency_ms": (
        rag_result.retrieval_latency_ms
    ),
    "generation_latency_ms": (
        rag_result.generation_latency_ms
    ),
    "total_latency_ms": (
        rag_result.total_latency_ms
    ),
}

print("Evaluation record prepared for Notebook 11.")
print(
    json.dumps(
        {
            "query": evaluation_record["query"],
            "report_year": evaluation_record["report_year"],
            "allowed_evidence_ids": (
                evaluation_record["allowed_evidence_ids"]
            ),
            "cited_evidence_ids": (
                evaluation_record["cited_evidence_ids"]
            ),
            "total_latency_ms": (
                evaluation_record["total_latency_ms"]
            ),
        },
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Completion

# COMMAND ----------

assert rag_result.answer.strip()
assert rag_result.citation_validation_passed
assert rag_result.answer_has_citations
assert rag_result.final_evidence_count > 0

print("10_baseline_rag COMPLETE")
print("")
print("Validated path:")
print(
    "query -> HYBRID AI Search -> context builder -> "
    "Llama generation -> citation validation -> MLflow"
)