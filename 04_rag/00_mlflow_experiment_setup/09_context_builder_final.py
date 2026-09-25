# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///


# COMMAND ----------

# Install the same AI Search package used successfully in Notebook 08.
%pip install -q --upgrade databricks-ai-search databricks-openai pydantic mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # 09_context_builder
# MAGIC
# MAGIC Purpose:
# MAGIC - Convert Notebook 08 retrieval evidence into a compact, citation-ready context package.
# MAGIC - Remove exact duplicate evidence and suppress strongly overlapping chunks.
# MAGIC - Preserve document/year/page provenance.
# MAGIC - Enforce a deterministic context budget.
# MAGIC - Keep retrieval evidence separate from the future answer-generation prompt.
# MAGIC
# MAGIC This notebook does NOT generate the final answer.
# MAGIC It prepares grounded evidence for Notebook 10.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Imports and configuration

# COMMAND ----------

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from pydantic import BaseModel, Field, field_validator

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_context_builder"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Notebook 08 returns up to six final evidence chunks.
MAX_EVIDENCE_ITEMS = 6

# Approximate context budget. We use a deterministic character/token estimate
# here rather than adding another model/tokenizer dependency.
MAX_CONTEXT_TOKENS = 6000

# Approximate English token conversion used only for budgeting.
CHARS_PER_TOKEN = 4.0

# Prevent a single retrieved chunk from consuming most of the context window.
MAX_TOKENS_PER_EVIDENCE = 1400

# Strong overlap threshold for near-duplicate evidence.
NEAR_DUPLICATE_JACCARD = 0.82

print("Context-builder configuration loaded.")
print("MAX_EVIDENCE_ITEMS:", MAX_EVIDENCE_ITEMS)
print("MAX_CONTEXT_TOKENS:", MAX_CONTEXT_TOKENS)
print("MAX_TOKENS_PER_EVIDENCE:", MAX_TOKENS_PER_EVIDENCE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Typed context contract
# MAGIC
# MAGIC Notebook 10 will consume `ContextPackage`.
# MAGIC Citation labels such as `[E1]` are internal evidence IDs, not fabricated
# MAGIC document citations. They map directly to report/year/page provenance.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Generic object conversion
# MAGIC
# MAGIC Notebook 08 may pass Pydantic models or dictionaries.
# MAGIC This keeps Notebook 09 independent from the exact class definition used there.

# COMMAND ----------

def to_dict(value: Any) -> Dict[str, Any]:
    """Convert Pydantic-like objects or dictionaries to a plain dictionary."""

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
    """Return the first non-null value from several possible field names."""

    for name in names:
        if name in row and row[name] is not None:
            return row[name]

    return default

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Text normalization and token budgeting

# COMMAND ----------

def normalize_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Deterministic token estimate used for context budgeting only.
    It is intentionally conservative and does not claim exact tokenizer counts.
    """

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
    """
    Truncate long evidence without rewriting it.

    We preserve source text and cut at a nearby word boundary.
    """

    text = normalize_whitespace(text)

    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = int(max_tokens * CHARS_PER_TOKEN)

    truncated = text[:max_chars]

    # Prefer a word boundary near the end.
    last_space = truncated.rfind(" ")

    if last_space >= int(max_chars * 0.85):
        truncated = truncated[:last_space]

    return truncated.rstrip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Duplicate and overlap detection
# MAGIC
# MAGIC Exact duplicates are detected from normalized source text.
# MAGIC Near duplicates use token-set Jaccard overlap and require the same document/year.
# MAGIC We keep the higher-ranked evidence and do not merge/rewrite source passages.

# COMMAND ----------

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
    """
    Suppress only strong overlaps.

    Requiring the same document/year plus either page overlap or parent identity
    reduces the risk of deleting genuinely different evidence that happens to
    discuss the same topic.
    """

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

# MAGIC %md
# MAGIC ## 06. Canonicalize Notebook 08 evidence
# MAGIC
# MAGIC We prefer `chunk_text` because it is the original source evidence.
# MAGIC `retrieval_text` is used only as a fallback if the retrieval object does not
# MAGIC expose `chunk_text`.

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

        document_id = get_first(
            row,
            ["document_id"],
        )

        report_year = get_first(
            row,
            ["report_year"],
        )

        page_start = get_first(
            row,
            ["page_start"],
        )

        page_end = get_first(
            row,
            ["page_end"],
            page_start,
        )

        # Provenance is mandatory for citation-ready context.
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
            [
                "retrieval_rank",
                "_retrieval_rank",
                "rank",
            ],
            position,
        )

        rerank_score = get_first(
            row,
            [
                "rerank_score",
                "_rerank_score",
            ],
        )

        canonical.append({
            "chunk_id": chunk_id,
            "parent_chunk_id": get_first(
                row,
                ["parent_chunk_id"],
            ),
            "document_id": str(document_id),
            "report_year": int(report_year),
            "edition_status": get_first(
                row,
                ["edition_status"],
            ),
            "chapter": get_first(
                row,
                ["chapter"],
            ),
            "region": get_first(
                row,
                ["region"],
            ),
            "section": get_first(
                row,
                ["section"],
            ),
            "subsection": get_first(
                row,
                ["subsection"],
            ),
            "content_type": get_first(
                row,
                ["content_type"],
            ),
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Rank-preserving deduplication

# COMMAND ----------

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

    return (
        selected,
        exact_removed,
        near_removed,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Context-budget selection
# MAGIC
# MAGIC We preserve retrieval/reranking order.
# MAGIC Evidence is never summarized or paraphrased in this layer.

# COMMAND ----------

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

        # Avoid including a tiny fragment simply because only a few tokens remain.
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

# MAGIC %md
# MAGIC ## 09. Citation-ready context formatting
# MAGIC
# MAGIC Each evidence block gets a stable local label `[E1]`, `[E2]`, etc.
# MAGIC The future generation model can cite these labels, and Notebook 10/guardrails
# MAGIC can validate that every cited label exists.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Main context-builder function

# COMMAND ----------

def build_context_package(
    retrieval_result: Any,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> ContextPackage:

    result = to_dict(retrieval_result)

    # Notebook 08 may call the final evidence field `evidence`,
    # `results`, or `final_evidence`. Support all three explicitly.
    raw_evidence = get_first(
        result,
        [
            "evidence",
            "final_evidence",
            "results",
        ],
        [],
    )

    if raw_evidence is None:
        raw_evidence = []

    if not isinstance(raw_evidence, list):
        raise TypeError(
            "Retrieval evidence must be a list."
        )

    if not raw_evidence:
        raise ValueError(
            "Cannot build grounded context from zero evidence chunks."
        )

    canonical = canonicalize_evidence(
        raw_evidence
    )

    if not canonical:
        raise ValueError(
            "No valid evidence remained after canonicalization."
        )

    (
        deduped,
        exact_removed,
        near_removed,
    ) = deduplicate_evidence(canonical)

    (
        selected,
        budget_drops,
    ) = select_under_budget(
        rows=deduped,
        max_context_tokens=max_context_tokens,
        max_items=max_items,
    )

    if not selected:
        raise ValueError(
            "No evidence fit inside the context budget."
        )

    evidence_models = []

    for i, row in enumerate(selected, start=1):

        evidence_models.append(
            EvidenceItem(
                evidence_id=f"E{i}",
                **row,
            )
        )

    context_text = build_context_text(
        selected
    )

    estimated_context_tokens = estimate_tokens(
        context_text
    )

    package = ContextPackage(
        original_query=str(
            get_first(
                result,
                ["original_query", "query"],
                "",
            )
        ),
        resolved_query=str(
            get_first(
                result,
                ["resolved_query", "search_query", "query"],
                "",
            )
        ),
        route_hint=str(
            get_first(
                result,
                ["route_hint"],
                "rag",
            )
        ),
        report_years=[
            int(x)
            for x in get_first(
                result,
                ["report_years"],
                [],
            )
        ],
        regions=[
            str(x)
            for x in get_first(
                result,
                ["regions"],
                [],
            )
        ],
        sections=[
            str(x)
            for x in get_first(
                result,
                ["sections"],
                [],
            )
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

    # Final invariants.
    assert package.final_evidence_count <= max_items
    assert package.final_evidence_count > 0

    # Formatting adds metadata tokens, so allow a modest envelope above the raw
    # evidence budget while still preventing uncontrolled context growth.
    assert package.estimated_context_tokens <= (
        max_context_tokens + 800
    ), (
        "Formatted context exceeded the expected budget envelope: "
        f"{package.estimated_context_tokens}"
    )

    return package

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Synthetic contract test
# MAGIC
# MAGIC This test validates context-building behavior only.
# MAGIC It does not pretend to be World Bank evidence.

# COMMAND ----------

synthetic_retrieval = {
    "original_query": "Example query",
    "resolved_query": "Example query",
    "route_hint": "rag",
    "report_years": [2025],
    "regions": ["South Asia"],
    "sections": ["Risks"],
    "evidence": [
        {
            "chunk_id": "c1",
            "parent_chunk_id": "p1",
            "document_id": "gep_2025_01",
            "report_year": 2025,
            "edition_status": "final",
            "region": "South Asia",
            "section": "Risks",
            "page_start": 80,
            "page_end": 81,
            "chunk_text": (
                "Example source passage used only to test "
                "the context-builder contract."
            ),
            "_retrieval_rank": 1,
        },
        {
            # Exact duplicate: should be removed.
            "chunk_id": "c2",
            "parent_chunk_id": "p1",
            "document_id": "gep_2025_01",
            "report_year": 2025,
            "edition_status": "final",
            "region": "South Asia",
            "section": "Risks",
            "page_start": 80,
            "page_end": 81,
            "chunk_text": (
                "Example source passage used only to test "
                "the context-builder contract."
            ),
            "_retrieval_rank": 2,
        },
        {
            "chunk_id": "c3",
            "parent_chunk_id": "p2",
            "document_id": "gep_2025_01",
            "report_year": 2025,
            "edition_status": "final",
            "region": "South Asia",
            "section": "Outlook",
            "page_start": 82,
            "page_end": 82,
            "chunk_text": (
                "A second distinct source passage verifies that "
                "multiple evidence blocks are preserved."
            ),
            "_retrieval_rank": 3,
        },
    ],
}

synthetic_package = build_context_package(
    synthetic_retrieval,
    max_context_tokens=1000,
)

print(
    synthetic_package.model_dump_json(indent=2)
)

assert synthetic_package.input_evidence_count == 3
assert synthetic_package.exact_duplicates_removed == 1
assert synthetic_package.final_evidence_count == 2
assert "[E1]" in synthetic_package.context_text
assert "[E2]" in synthetic_package.context_text

print("Synthetic context-builder contract test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Real AI Search integration test
# MAGIC
# MAGIC Notebook 09 must be independently runnable.
# MAGIC It does NOT `%run` Notebook 08 and does NOT depend on Notebook 08's Python memory.
# MAGIC
# MAGIC For this integration test we query the same production-candidate AI Search index
# MAGIC directly, using the retrieval configuration already validated in Notebooks 05-08.
# MAGIC This tests the boundary that matters here:
# MAGIC
# MAGIC real AI Search evidence -> context builder -> citation-ready context

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
AI_SEARCH_INDEX = "worldbank_ai.rag.gep_structure_v1_qwen3_index"

REAL_TEST_QUERY = (
    "What risks did the World Bank identify "
    "for South Asia in 2025?"
)

REAL_TEST_YEAR = 2025
REAL_RETRIEVAL_TOP_K = 6

# Columns required by the context builder.
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

real_index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=AI_SEARCH_INDEX,
)

index_description = real_index.describe()

print("AI Search endpoint:", AI_SEARCH_ENDPOINT)
print("AI Search index:", AI_SEARCH_INDEX)
print("Test query:", REAL_TEST_QUERY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Execute real HYBRID retrieval
# MAGIC
# MAGIC We apply the high-confidence report-year filter.
# MAGIC Region/section are intentionally not hard-filtered here because upstream metadata
# MAGIC is incomplete for some chunks.

# COMMAND ----------

real_search_response = real_index.similarity_search(
    query_text=REAL_TEST_QUERY,
    columns=RETRIEVAL_COLUMNS,
    num_results=REAL_RETRIEVAL_TOP_K,
    query_type="HYBRID",
    filters={
        "report_year": REAL_TEST_YEAR,
    },
)

print("Real AI Search request completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Normalize Databricks AI Search response
# MAGIC
# MAGIC AI Search returns a manifest describing column order plus a data array.
# MAGIC We convert those rows into the evidence dictionaries expected by the context builder.
# MAGIC No fake evidence and no silent fallback are used.

# COMMAND ----------

def normalize_ai_search_response(response):
    """Convert Databricks AI Search response rows into dictionaries."""

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


real_evidence_rows = normalize_ai_search_response(
    real_search_response
)

print(
    "Real retrieved evidence chunks:",
    len(real_evidence_rows),
)

assert len(real_evidence_rows) > 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Validate real retrieval provenance before context construction

# COMMAND ----------

required_fields = [
    "chunk_id",
    "document_id",
    "report_year",
    "page_start",
    "page_end",
    "chunk_text",
]

for rank, row in enumerate(real_evidence_rows, start=1):

    missing = [
        field
        for field in required_fields
        if row.get(field) is None
    ]

    if missing:
        raise ValueError(
            f"Retrieved evidence rank {rank} is missing: {missing}"
        )

    if int(row["report_year"]) != REAL_TEST_YEAR:
        raise AssertionError(
            "Year-filtered AI Search returned an unexpected "
            f"report year at rank {rank}: {row['report_year']}"
        )

    if not str(row["chunk_text"]).strip():
        raise ValueError(
            f"Retrieved evidence rank {rank} has empty chunk_text."
        )

print("Real retrieval provenance validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Create the retrieval contract consumed by the context builder

# COMMAND ----------

real_retrieval_result = {
    "original_query": REAL_TEST_QUERY,
    "resolved_query": REAL_TEST_QUERY,
    "route_hint": "rag",
    "report_years": [REAL_TEST_YEAR],
    "regions": ["South Asia"],
    "sections": ["Risks"],
    "evidence": real_evidence_rows,
}

print(
    "Retrieval contract created with",
    len(real_retrieval_result["evidence"]),
    "real evidence chunks.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Build citation-ready context from real World Bank evidence

# COMMAND ----------

real_context_package = build_context_package(
    real_retrieval_result
)

print("REAL CONTEXT PACKAGE")
print("--------------------")

print(
    real_context_package.model_dump_json(
        indent=2,
        exclude={"context_text"},
    )
)

print("")
print("Context statistics")
print("------------------")
print("Input evidence:", real_context_package.input_evidence_count)
print("Final evidence:", real_context_package.final_evidence_count)
print("Exact duplicates removed:", real_context_package.exact_duplicates_removed)
print("Near duplicates removed:", real_context_package.near_duplicates_removed)
print("Budget drops:", real_context_package.budget_drops)
print("Estimated context tokens:", real_context_package.estimated_context_tokens)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Inspect the real citation-ready evidence

# COMMAND ----------

for item in real_context_package.evidence:

    print("=" * 100)

    print(
        f"{item.evidence_id} | "
        f"Year={item.report_year} | "
        f"Pages={item.page_start}-{item.page_end}"
    )

    print("Document:", item.document_id)
    print("Region:", item.region)
    print("Section:", item.section)
    print("Chunk:", item.chunk_id)
    print("Estimated tokens:", item.estimated_tokens)

    print("")
    print(item.text[:1000])
    print("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Validate real context provenance

# COMMAND ----------

assert real_context_package.final_evidence_count > 0

assert (
    real_context_package.final_evidence_count
    <= MAX_EVIDENCE_ITEMS
)

evidence_ids = [
    item.evidence_id
    for item in real_context_package.evidence
]

assert len(evidence_ids) == len(set(evidence_ids))

chunk_ids = [
    item.chunk_id
    for item in real_context_package.evidence
]

assert len(chunk_ids) == len(set(chunk_ids))

for item in real_context_package.evidence:

    assert item.document_id
    assert item.chunk_id
    assert item.report_year == REAL_TEST_YEAR
    assert item.page_start >= 1
    assert item.page_end >= item.page_start
    assert item.text.strip()

    assert (
        f"[{item.evidence_id}]"
        in real_context_package.context_text
    )

print("Real context provenance validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Validate context budget

# COMMAND ----------

CONTEXT_FORMATTING_ALLOWANCE = 800

print(
    "Estimated context tokens:",
    real_context_package.estimated_context_tokens,
)

print(
    "Configured evidence budget:",
    MAX_CONTEXT_TOKENS,
)

assert (
    real_context_package.estimated_context_tokens
    <= MAX_CONTEXT_TOKENS
    + CONTEXT_FORMATTING_ALLOWANCE
)

print("Context budget validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Build Notebook 10 generation contract

# COMMAND ----------

generation_input = {
    "query": real_context_package.resolved_query,
    "context": real_context_package.context_text,
    "allowed_evidence_ids": [
        item.evidence_id
        for item in real_context_package.evidence
    ],
    "evidence_metadata": [
        {
            "evidence_id": item.evidence_id,
            "document_id": item.document_id,
            "report_year": item.report_year,
            "edition_status": item.edition_status,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "chunk_id": item.chunk_id,
        }
        for item in real_context_package.evidence
    ],
}

print(
    json.dumps(
        {
            "query": generation_input["query"],
            "allowed_evidence_ids": generation_input[
                "allowed_evidence_ids"
            ],
            "context_chars": len(
                generation_input["context"]
            ),
            "evidence_count": len(
                generation_input["evidence_metadata"]
            ),
        },
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. MLflow real-context smoke run

# COMMAND ----------

with mlflow.start_run(
    run_name="context_builder_real_ai_search_v1"
):

    mlflow.log_params({
        "test_query": REAL_TEST_QUERY,
        "ai_search_endpoint": AI_SEARCH_ENDPOINT,
        "ai_search_index": AI_SEARCH_INDEX,
        "retrieval_type": "HYBRID",
        "report_year_filter": REAL_TEST_YEAR,
        "retrieval_top_k": REAL_RETRIEVAL_TOP_K,
        "strategy": "rank_preserving_dedup_budget",
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_tokens_per_evidence": MAX_TOKENS_PER_EVIDENCE,
        "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
    })

    mlflow.log_metrics({
        "input_evidence_count": (
            real_context_package.input_evidence_count
        ),
        "final_evidence_count": (
            real_context_package.final_evidence_count
        ),
        "exact_duplicates_removed": (
            real_context_package.exact_duplicates_removed
        ),
        "near_duplicates_removed": (
            real_context_package.near_duplicates_removed
        ),
        "budget_drops": (
            real_context_package.budget_drops
        ),
        "estimated_context_tokens": (
            real_context_package.estimated_context_tokens
        ),
    })

print("MLflow real context-builder run logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. Completion

# COMMAND ----------

print("09_context_builder COMPLETE")
print("")
print("Validated with REAL Databricks AI Search retrieval:")
print("- Structure V1 Qwen3 index")
print("- HYBRID retrieval")
print("- 2025 report-year metadata filter")
print("- real World Bank GEP evidence")
print("- exact duplicate removal")
print("- conservative near-duplicate suppression")
print("- deterministic context budget")
print("- document/year/page/chunk provenance")
print("- stable [E#] evidence labels")
print("- generation contract")
print("- MLflow logging")
print("")
print("Ready for: 10_baseline_rag")