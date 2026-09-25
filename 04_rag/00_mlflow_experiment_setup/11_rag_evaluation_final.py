# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 11_rag_evaluation
# MAGIC
# MAGIC Evaluate the locked `baseline_v1` RAG pipeline on the governed 50-question
# MAGIC Global Economic Prospects benchmark.
# MAGIC
# MAGIC Evaluation layers:
# MAGIC
# MAGIC 1. Retrieval — gold page/chunk hit, reciprocal rank
# MAGIC 2. Generation — groundedness, relevance, reference-evidence coverage
# MAGIC 3. Citations — validity, coverage, unsupported citation rate
# MAGIC 4. System — retrieval/generation/end-to-end latency and failure rate
# MAGIC 5. Failure analysis — retrieval, grounding, relevance, citation, runtime failures
# MAGIC
# MAGIC Important:
# MAGIC - No fake scores.
# MAGIC - No synthetic benchmark rows.
# MAGIC - The governed 50-question Delta benchmark is the source of truth.
# MAGIC - Notebook 05-10 remain unchanged.
# MAGIC - `reference_evidence` is treated as source evidence, NOT as a word-for-word reference answer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Imports and configuration
# MAGIC
# MAGIC This notebook assumes the packages used by Notebook 10 are already available.
# MAGIC If `databricks.ai_search` is missing after an environment reset, install
# MAGIC `databricks-ai-search` and `databricks-openai`, restart Python once, then rerun.

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from pyspark.sql import functions as F
from pyspark.sql import types as T

from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI

CATALOG = "worldbank_ai"
RAG_SCHEMA = "rag"
AI_SCHEMA = "ai"

EVAL_DATASET = f"{CATALOG}.{RAG_SCHEMA}.retrieval_eval_dataset"
RESULTS_TABLE = f"{CATALOG}.{AI_SCHEMA}.rag_evaluation_results"
SUMMARY_TABLE = f"{CATALOG}.{AI_SCHEMA}.rag_evaluation_summary"

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
AI_SEARCH_INDEX = f"{CATALOG}.{RAG_SCHEMA}.gep_structure_v1_qwen3_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_rag_evaluation"
PIPELINE_VERSION = "baseline_v1"
EVALUATION_VERSION = "rag_eval_v1"

# Locked Notebook 10 baseline settings.
RETRIEVAL_TOP_K = 6
QUERY_TYPE = "HYBRID"
MAX_EVIDENCE_ITEMS = 6
MAX_CONTEXT_TOKENS = 6000
CHARS_PER_TOKEN = 4.0
MAX_TOKENS_PER_EVIDENCE = 1400
NEAR_DUPLICATE_JACCARD = 0.82
MAX_ANSWER_TOKENS = 1200

# Judge configuration.
# Same workspace endpoint, but a separate call from answer generation.
JUDGE_MODEL = LLM_ENDPOINT
JUDGE_MAX_TOKENS = 700

# During debugging you can temporarily set this to 3 or 5.
# For the final experiment leave it as None to evaluate all 50.
DEBUG_LIMIT = None

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{AI_SCHEMA}")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print("11_rag_evaluation configuration loaded.")
print("Benchmark:", EVAL_DATASET)
print("Results:", RESULTS_TABLE)
print("Summary:", SUMMARY_TABLE)
print("Pipeline version:", PIPELINE_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Validate the governed 50-question benchmark

# COMMAND ----------

assert spark.catalog.tableExists(EVAL_DATASET), f"Missing benchmark: {EVAL_DATASET}"

eval_df = spark.table(EVAL_DATASET)

required_columns = {
    "eval_id",
    "question",
    "question_type",
    "difficulty",
    "expected_report_year",
    "relevant_document_id",
    "relevant_page_start",
    "relevant_page_end",
    "relevant_chunk_id",
    "reference_evidence",
}

missing = sorted(required_columns - set(eval_df.columns))
assert not missing, f"Benchmark missing required columns: {missing}"

approved_count = eval_df.count()
assert approved_count == 50, f"Expected 50 benchmark rows, found {approved_count}"
assert eval_df.select("eval_id").distinct().count() == 50
assert eval_df.select("question").distinct().count() == 50
assert eval_df.filter(F.col("reference_evidence").isNull()).count() == 0
assert eval_df.filter(F.col("relevant_chunk_id").isNull()).count() == 0

print("Benchmark validation passed: 50 governed questions.")

display(
    eval_df.groupBy("expected_report_year")
           .count()
           .orderBy("expected_report_year")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Load evaluation rows
# MAGIC
# MAGIC Collection is acceptable here because the governed benchmark contains only 50 rows.
# MAGIC We preserve the benchmark's existing order using `eval_id`.

# COMMAND ----------

ordered_eval_df = eval_df.orderBy("eval_id")

if DEBUG_LIMIT is not None:
    ordered_eval_df = ordered_eval_df.limit(int(DEBUG_LIMIT))

benchmark_rows = [row.asDict(recursive=True) for row in ordered_eval_df.collect()]

print(f"Rows scheduled for evaluation: {len(benchmark_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Connect to AI Search and Model Serving

# COMMAND ----------

ai_search_client = AISearchClient()
search_index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=AI_SEARCH_INDEX,
)

index_description = search_index.describe()
index_status = index_description.get("status", {})

print("AI Search status:", index_status)

llm_client = DatabricksOpenAI()

print("AI Search and LLM clients initialized.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. Shared text/context helpers
# MAGIC
# MAGIC These preserve the same context-building policy used by the locked baseline.

# COMMAND ----------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / CHARS_PER_TOKEN)))


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
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
    return re.sub(r"\s+", " ", text).strip()


def lexical_token_set(text: str) -> set:
    return {
        token
        for token in normalized_fingerprint(text).split()
        if len(token) >= 3
    }


def jaccard_similarity(left: str, right: str) -> float:
    a = lexical_token_set(left)
    b = lexical_token_set(right)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def pages_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(int(a_start), int(b_start)) <= min(int(a_end), int(b_end))


def is_near_duplicate(candidate: Dict[str, Any], selected: Dict[str, Any]) -> bool:
    if str(candidate["document_id"]) != str(selected["document_id"]):
        return False

    if int(candidate["report_year"]) != int(selected["report_year"]):
        return False

    same_parent = (
        candidate.get("parent_chunk_id")
        and selected.get("parent_chunk_id")
        and str(candidate["parent_chunk_id"]) == str(selected["parent_chunk_id"])
    )

    overlapping_pages = pages_overlap(
        candidate["page_start"],
        candidate["page_end"],
        selected["page_start"],
        selected["page_end"],
    )

    if not (same_parent or overlapping_pages):
        return False

    return (
        jaccard_similarity(candidate["chunk_text"], selected["chunk_text"])
        >= NEAR_DUPLICATE_JACCARD
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. AI Search response parser

# COMMAND ----------

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
]


def parse_search_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    if response is None:
        raise ValueError("AI Search returned None.")

    manifest_columns = response.get("manifest", {}).get("columns", [])
    rows = response.get("result", {}).get("data_array", [])

    if not manifest_columns:
        raise ValueError("AI Search response is missing manifest columns.")

    names = []
    for col in manifest_columns:
        if isinstance(col, dict):
            names.append(col.get("name"))
        else:
            names.append(str(col))

    parsed = []

    for rank, row in enumerate(rows, start=1):
        # AI Search may append a score after the manifest columns.
        values = row[:len(names)]
        score = row[len(names)] if len(row) > len(names) else None

        item = dict(zip(names, values))
        item["_retrieval_rank"] = rank
        item["_score"] = score
        parsed.append(item)

    return parsed

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Build the locked baseline context

# COMMAND ----------

def deduplicate_and_budget(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    exact_seen = set()

    for row in rows:
        text = normalize_whitespace(str(row.get("chunk_text") or ""))

        if not text:
            continue

        canonical = dict(row)
        canonical["chunk_text"] = text

        fingerprint = (
            str(canonical["document_id"]),
            int(canonical["report_year"]),
            normalized_fingerprint(text),
        )

        if fingerprint in exact_seen:
            continue

        if any(is_near_duplicate(canonical, prior) for prior in selected):
            continue

        exact_seen.add(fingerprint)
        selected.append(canonical)

    final_rows = []
    used_tokens = 0

    for row in selected:
        if len(final_rows) >= MAX_EVIDENCE_ITEMS:
            break

        remaining = MAX_CONTEXT_TOKENS - used_tokens

        if remaining <= 0:
            break

        per_item_budget = min(MAX_TOKENS_PER_EVIDENCE, remaining)

        if per_item_budget < 80:
            break

        text = truncate_to_token_budget(row["chunk_text"], per_item_budget)

        if not text:
            continue

        item = dict(row)
        item["chunk_text"] = text
        item["estimated_tokens"] = estimate_tokens(text)

        final_rows.append(item)
        used_tokens += item["estimated_tokens"]

    if not final_rows:
        raise ValueError("No evidence survived context construction.")

    return final_rows


def format_pages(start: int, end: int) -> str:
    return f"p. {start}" if int(start) == int(end) else f"pp. {start}-{end}"


def build_context_text(rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    blocks = []
    evidence = []

    for i, row in enumerate(rows, start=1):
        evidence_id = f"E{i}"

        metadata = [
            f"report_year={row['report_year']}",
            f"pages={format_pages(row['page_start'], row['page_end'])}",
            f"document_id={row['document_id']}",
            f"chunk_id={row['chunk_id']}",
        ]

        for field in ["edition_status", "chapter", "region", "section", "subsection"]:
            value = row.get(field)
            if value is not None and str(value).strip():
                metadata.append(f"{field}={str(value).strip()}")

        blocks.append(
            f"[{evidence_id}]\n"
            + " | ".join(metadata)
            + "\n"
            + row["chunk_text"]
        )

        enriched = dict(row)
        enriched["evidence_id"] = evidence_id
        evidence.append(enriched)

    return "\n\n---\n\n".join(blocks), evidence

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Locked baseline generation prompt

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


def generate_answer(question: str, context_text: str) -> Tuple[str, float]:
    user_prompt = f"""
QUESTION
{question}

RETRIEVED EVIDENCE
{context_text}

Write a grounded answer to the QUESTION using only the RETRIEVED EVIDENCE.
""".strip()

    started = time.perf_counter()

    response = llm_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    latency_ms = (time.perf_counter() - started) * 1000.0
    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        raise ValueError("Generation returned an empty answer.")

    return answer, latency_ms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Deterministic retrieval and citation metrics
# MAGIC
# MAGIC Retrieval ground truth is strategy-neutral:
# MAGIC correct document + overlapping gold page range.
# MAGIC Exact gold chunk hit is also recorded separately.

# COMMAND ----------

def gold_rank(
    retrieved: List[Dict[str, Any]],
    gold_document_id: str,
    gold_page_start: int,
    gold_page_end: int,
) -> Optional[int]:
    for rank, row in enumerate(retrieved, start=1):
        same_document = str(row.get("document_id")) == str(gold_document_id)

        if row.get("page_start") is None or row.get("page_end") is None:
            continue

        overlap = pages_overlap(
            row["page_start"],
            row["page_end"],
            gold_page_start,
            gold_page_end,
        )

        if same_document and overlap:
            return rank

    return None


def exact_gold_chunk_rank(
    retrieved: List[Dict[str, Any]],
    relevant_chunk_id: str,
) -> Optional[int]:
    for rank, row in enumerate(retrieved, start=1):
        if str(row.get("chunk_id")) == str(relevant_chunk_id):
            return rank
    return None


def extract_citations(answer: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"\[(E\d+)\]", answer or "")))


def citation_metrics(
    answer: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    allowed = {row["evidence_id"] for row in evidence}
    cited = extract_citations(answer)
    cited_set = set(cited)

    unsupported = sorted(cited_set - allowed)
    valid = sorted(cited_set & allowed)

    return {
        "cited_evidence_ids": cited,
        "citation_count": len(cited),
        "valid_citation_count": len(valid),
        "unsupported_citation_count": len(unsupported),
        "unsupported_citation_rate": (
            len(unsupported) / len(cited) if cited else 0.0
        ),
        "citation_validity_pass": bool(cited) and not unsupported,
        "answer_has_citations": bool(cited),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. LLM quality judge
# MAGIC
# MAGIC We deliberately use a structured judge because the benchmark contains
# MAGIC `reference_evidence`, not a handcrafted reference answer.
# MAGIC
# MAGIC The judge scores:
# MAGIC - groundedness: are answer claims supported by retrieved context?
# MAGIC - relevance: does the answer directly answer the question?
# MAGIC - reference coverage: does the answer capture the important information
# MAGIC   in the governed gold evidence that is relevant to the question?
# MAGIC
# MAGIC Scores are 1-5. They are kept separate from deterministic metrics.

# COMMAND ----------

JUDGE_SYSTEM_PROMPT = """
You are evaluating a retrieval-augmented generation system for World Bank Global Economic Prospects reports.

Evaluate ONLY from the material supplied to you.
Do not use outside knowledge.

Return exactly one valid JSON object and no markdown.

Scoring:
1 = very poor
2 = poor
3 = acceptable but incomplete
4 = good
5 = excellent

Fields:
- groundedness_score: Are factual claims in the answer supported by RETRIEVED_CONTEXT?
- relevance_score: Does the answer directly address QUESTION?
- reference_coverage_score: Does the answer capture the important information in GOLD_REFERENCE_EVIDENCE that is relevant to QUESTION?
- groundedness_reason: short reason
- relevance_reason: short reason
- reference_coverage_reason: short reason
- hallucination_detected: true if the answer contains a factual claim not supported by RETRIEVED_CONTEXT
- major_omission_detected: true if an important answerable point from GOLD_REFERENCE_EVIDENCE is missing
""".strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # Remove accidental fenced formatting defensively.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_answer(
    question: str,
    answer: str,
    retrieved_context: str,
    reference_evidence: str,
) -> Tuple[Dict[str, Any], float]:
    prompt = f"""
QUESTION:
{question}

ANSWER:
{answer}

RETRIEVED_CONTEXT:
{retrieved_context}

GOLD_REFERENCE_EVIDENCE:
{reference_evidence}
""".strip()

    started = time.perf_counter()

    response = llm_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    latency_ms = (time.perf_counter() - started) * 1000.0
    raw = response.choices[0].message.content or ""
    result = parse_json_object(raw)

    required = [
        "groundedness_score",
        "relevance_score",
        "reference_coverage_score",
        "groundedness_reason",
        "relevance_reason",
        "reference_coverage_reason",
        "hallucination_detected",
        "major_omission_detected",
    ]

    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"Judge response missing fields: {missing}")

    for score_name in [
        "groundedness_score",
        "relevance_score",
        "reference_coverage_score",
    ]:
        score = float(result[score_name])
        if score < 1 or score > 5:
            raise ValueError(f"{score_name} outside 1-5: {score}")
        result[score_name] = score

    result["hallucination_detected"] = bool(result["hallucination_detected"])
    result["major_omission_detected"] = bool(result["major_omission_detected"])

    return result, latency_ms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Evaluate one benchmark row
# MAGIC
# MAGIC Keeping this as a single function makes failure handling explicit and lets
# MAGIC us persist every case, including failures.

# COMMAND ----------

def evaluate_one_case(case: Dict[str, Any]) -> Dict[str, Any]:
    eval_id = str(case["eval_id"])
    question = str(case["question"])
    expected_year = int(case["expected_report_year"])

    result = {
        "evaluation_version": EVALUATION_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "eval_id": eval_id,
        "question": question,
        "question_type": case.get("question_type"),
        "difficulty": case.get("difficulty"),
        "expected_report_year": expected_year,
        "relevant_document_id": str(case["relevant_document_id"]),
        "relevant_page_start": int(case["relevant_page_start"]),
        "relevant_page_end": int(case["relevant_page_end"]),
        "relevant_chunk_id": str(case["relevant_chunk_id"]),
        "status": "FAILED",
        "error_type": None,
        "error_message": None,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    total_started = time.perf_counter()

    try:
        # ---------------- Retrieval ----------------
        retrieval_started = time.perf_counter()

        response = search_index.similarity_search(
            query_text=question,
            columns=RETURN_COLUMNS,
            num_results=RETRIEVAL_TOP_K,
            query_type=QUERY_TYPE,
            filters={"report_year": expected_year},
        )

        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
        retrieved = parse_search_results(response)

        if not retrieved:
            raise ValueError("AI Search returned zero rows.")

        # ---------------- Retrieval metrics ----------------
        rank = gold_rank(
            retrieved,
            case["relevant_document_id"],
            case["relevant_page_start"],
            case["relevant_page_end"],
        )

        chunk_rank = exact_gold_chunk_rank(
            retrieved,
            case["relevant_chunk_id"],
        )

        # ---------------- Context ----------------
        context_rows = deduplicate_and_budget(retrieved)
        context_text, evidence = build_context_text(context_rows)

        # ---------------- Generation ----------------
        answer, generation_latency_ms = generate_answer(
            question,
            context_text,
        )

        # ---------------- Citations ----------------
        citations = citation_metrics(answer, evidence)

        # Did the answer cite at least one evidence item that overlaps the gold pages?
        gold_evidence_ids = {
            row["evidence_id"]
            for row in evidence
            if (
                str(row["document_id"]) == str(case["relevant_document_id"])
                and pages_overlap(
                    row["page_start"],
                    row["page_end"],
                    case["relevant_page_start"],
                    case["relevant_page_end"],
                )
            )
        }

        cited_set = set(citations["cited_evidence_ids"])
        gold_evidence_cited = bool(gold_evidence_ids & cited_set)

        # ---------------- LLM judge ----------------
        judge_result, judge_latency_ms = judge_answer(
            question=question,
            answer=answer,
            retrieved_context=context_text,
            reference_evidence=str(case["reference_evidence"]),
        )

        total_latency_ms = (time.perf_counter() - total_started) * 1000.0

        result.update({
            "status": "PASSED",
            "retrieved_count": len(retrieved),
            "context_evidence_count": len(evidence),
            "estimated_context_tokens": estimate_tokens(context_text),

            "gold_rank": rank,
            "exact_gold_chunk_rank": chunk_rank,
            "hit_at_1": bool(rank is not None and rank <= 1),
            "hit_at_3": bool(rank is not None and rank <= 3),
            "hit_at_5": bool(rank is not None and rank <= 5),
            "hit_at_6": bool(rank is not None and rank <= 6),
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            "exact_gold_chunk_hit": chunk_rank is not None,

            "answer": answer,
            "context_text": context_text,

            "cited_evidence_ids_json": json.dumps(citations["cited_evidence_ids"]),
            "citation_count": citations["citation_count"],
            "valid_citation_count": citations["valid_citation_count"],
            "unsupported_citation_count": citations["unsupported_citation_count"],
            "unsupported_citation_rate": citations["unsupported_citation_rate"],
            "citation_validity_pass": citations["citation_validity_pass"],
            "answer_has_citations": citations["answer_has_citations"],
            "gold_evidence_cited": gold_evidence_cited,

            "groundedness_score": judge_result["groundedness_score"],
            "relevance_score": judge_result["relevance_score"],
            "reference_coverage_score": judge_result["reference_coverage_score"],
            "groundedness_reason": str(judge_result["groundedness_reason"]),
            "relevance_reason": str(judge_result["relevance_reason"]),
            "reference_coverage_reason": str(judge_result["reference_coverage_reason"]),
            "hallucination_detected": judge_result["hallucination_detected"],
            "major_omission_detected": judge_result["major_omission_detected"],

            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "judge_latency_ms": judge_latency_ms,
            "total_latency_ms": total_latency_ms,
        })

    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:4000]
        result["total_latency_ms"] = (time.perf_counter() - total_started) * 1000.0

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Run the 50-question evaluation
# MAGIC
# MAGIC This makes roughly:
# MAGIC - 50 AI Search calls
# MAGIC - 50 answer-generation calls
# MAGIC - 50 judge calls
# MAGIC
# MAGIC so it can take several minutes. Progress is printed after every case.

# COMMAND ----------

evaluation_results = []

with mlflow.start_run(run_name=f"{PIPELINE_VERSION}_{EVALUATION_VERSION}") as active_run:
    evaluation_run_id = active_run.info.run_id

    mlflow.log_params({
        "pipeline_version": PIPELINE_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "benchmark_table": EVAL_DATASET,
        "benchmark_rows": len(benchmark_rows),
        "ai_search_index": AI_SEARCH_INDEX,
        "query_type": QUERY_TYPE,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "generation_model": LLM_ENDPOINT,
        "judge_model": JUDGE_MODEL,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
    })

    for i, case in enumerate(benchmark_rows, start=1):
        print(
            f"[{i:02d}/{len(benchmark_rows):02d}] "
            f"{case['eval_id']} | year={case['expected_report_year']}"
        )

        row_result = evaluate_one_case(case)
        row_result["mlflow_run_id"] = evaluation_run_id
        evaluation_results.append(row_result)

        if row_result["status"] == "PASSED":
            print(
                "  PASS | "
                f"Hit@5={row_result['hit_at_5']} | "
                f"grounded={row_result['groundedness_score']:.1f} | "
                f"relevance={row_result['relevance_score']:.1f} | "
                f"coverage={row_result['reference_coverage_score']:.1f} | "
                f"latency={row_result['total_latency_ms']:.0f} ms"
            )
        else:
            print(
                "  FAIL | "
                f"{row_result['error_type']}: {row_result['error_message']}"
            )

print("Evaluation loop complete.")
print("MLflow run ID:", evaluation_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Persist per-question evaluation results

# COMMAND ----------

# ============================================================
# 13. Persist per-question evaluation results
# ============================================================
# Use an explicit schema instead of asking Spark to infer types.
#
# Why:
# Some fields can legitimately be None for every row in a small
# debug run. Spark cannot infer a datatype from all-None values.

from pyspark.sql import types as T

assert evaluation_results, "No evaluation results were produced."


# ------------------------------------------------------------
# Explicit schema for one evaluation result
# ------------------------------------------------------------

evaluation_results_schema = T.StructType([

    # Experiment identity
    T.StructField("evaluation_version", T.StringType(), False),
    T.StructField("pipeline_version", T.StringType(), False),
    T.StructField("mlflow_run_id", T.StringType(), True),

    # Benchmark identity
    T.StructField("eval_id", T.StringType(), False),
    T.StructField("question", T.StringType(), False),
    T.StructField("question_type", T.StringType(), True),
    T.StructField("difficulty", T.StringType(), True),

    # Gold provenance
    T.StructField("expected_report_year", T.IntegerType(), True),
    T.StructField("relevant_document_id", T.StringType(), True),
    T.StructField("relevant_page_start", T.IntegerType(), True),
    T.StructField("relevant_page_end", T.IntegerType(), True),
    T.StructField("relevant_chunk_id", T.StringType(), True),

    # Execution status
    T.StructField("status", T.StringType(), False),
    T.StructField("error_type", T.StringType(), True),
    T.StructField("error_message", T.StringType(), True),
    T.StructField("evaluated_at_utc", T.StringType(), True),

    # Retrieval/context
    T.StructField("retrieved_count", T.IntegerType(), True),
    T.StructField("context_evidence_count", T.IntegerType(), True),
    T.StructField("estimated_context_tokens", T.IntegerType(), True),

    # Retrieval evaluation
    T.StructField("gold_rank", T.IntegerType(), True),
    T.StructField("exact_gold_chunk_rank", T.IntegerType(), True),

    T.StructField("hit_at_1", T.BooleanType(), True),
    T.StructField("hit_at_3", T.BooleanType(), True),
    T.StructField("hit_at_5", T.BooleanType(), True),
    T.StructField("hit_at_6", T.BooleanType(), True),

    T.StructField("reciprocal_rank", T.DoubleType(), True),
    T.StructField("exact_gold_chunk_hit", T.BooleanType(), True),

    # Generated answer/context
    T.StructField("answer", T.StringType(), True),
    T.StructField("context_text", T.StringType(), True),

    # Citation evaluation
    T.StructField("cited_evidence_ids_json", T.StringType(), True),
    T.StructField("citation_count", T.IntegerType(), True),
    T.StructField("valid_citation_count", T.IntegerType(), True),
    T.StructField("unsupported_citation_count", T.IntegerType(), True),
    T.StructField("unsupported_citation_rate", T.DoubleType(), True),

    T.StructField("citation_validity_pass", T.BooleanType(), True),
    T.StructField("answer_has_citations", T.BooleanType(), True),
    T.StructField("gold_evidence_cited", T.BooleanType(), True),

    # LLM judge evaluation
    T.StructField("groundedness_score", T.DoubleType(), True),
    T.StructField("relevance_score", T.DoubleType(), True),
    T.StructField("reference_coverage_score", T.DoubleType(), True),

    T.StructField("groundedness_reason", T.StringType(), True),
    T.StructField("relevance_reason", T.StringType(), True),
    T.StructField("reference_coverage_reason", T.StringType(), True),

    T.StructField("hallucination_detected", T.BooleanType(), True),
    T.StructField("major_omission_detected", T.BooleanType(), True),

    # Latency
    T.StructField("retrieval_latency_ms", T.DoubleType(), True),
    T.StructField("generation_latency_ms", T.DoubleType(), True),
    T.StructField("judge_latency_ms", T.DoubleType(), True),
    T.StructField("total_latency_ms", T.DoubleType(), True),
])


# ------------------------------------------------------------
# Normalize every result to exactly the schema above.
#
# Failed cases may not contain later-stage fields such as
# groundedness_score or generation_latency_ms.
# Missing dictionary keys therefore become None.
# ------------------------------------------------------------

schema_fields = [field.name for field in evaluation_results_schema.fields]

normalized_results = []

for result in evaluation_results:
    normalized_results.append({
        field_name: result.get(field_name)
        for field_name in schema_fields
    })


# ------------------------------------------------------------
# Create Spark DataFrame using the explicit schema
# ------------------------------------------------------------

results_df = spark.createDataFrame(
    normalized_results,
    schema=evaluation_results_schema
)


# ------------------------------------------------------------
# Validate before writing
# ------------------------------------------------------------

print("Rows to persist:", results_df.count())

results_df.printSchema()

display(
    results_df.select(
        "eval_id",
        "status",
        "gold_rank",
        "hit_at_5",
        "groundedness_score",
        "relevance_score",
        "reference_coverage_score",
        "citation_validity_pass",
        "total_latency_ms",
        "error_type",
    )
)


# ------------------------------------------------------------
# Persist to Delta
# ------------------------------------------------------------

(
    results_df.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(RESULTS_TABLE)
)

print(
    f"Persisted {results_df.count()} rows "
    f"to {RESULTS_TABLE}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Aggregate metrics
# MAGIC
# MAGIC Metrics are calculated only over successfully executed cases.
# MAGIC Failure rate is calculated over all scheduled benchmark cases.

# COMMAND ----------

run_df = (
    spark.table(RESULTS_TABLE)
    .filter(F.col("mlflow_run_id") == evaluation_run_id)
)

total_cases = run_df.count()
passed_cases = run_df.filter(F.col("status") == "PASSED").count()
failed_cases = total_cases - passed_cases

assert total_cases == len(benchmark_rows)

passed_df = run_df.filter(F.col("status") == "PASSED")

if passed_cases == 0:
    raise RuntimeError("All evaluation cases failed; inspect per-question errors.")

metrics_row = (
    passed_df.agg(
        F.avg(F.col("hit_at_1").cast("double")).alias("hit_at_1"),
        F.avg(F.col("hit_at_3").cast("double")).alias("hit_at_3"),
        F.avg(F.col("hit_at_5").cast("double")).alias("hit_at_5"),
        F.avg(F.col("hit_at_6").cast("double")).alias("hit_at_6"),
        F.avg("reciprocal_rank").alias("mrr_at_6"),
        F.avg(F.col("exact_gold_chunk_hit").cast("double")).alias("exact_gold_chunk_hit_rate"),

        F.avg(F.col("citation_validity_pass").cast("double")).alias("citation_validity_rate"),
        F.avg(F.col("answer_has_citations").cast("double")).alias("answer_citation_rate"),
        F.avg(F.col("gold_evidence_cited").cast("double")).alias("gold_evidence_citation_rate"),
        F.avg("unsupported_citation_rate").alias("unsupported_citation_rate"),

        F.avg("groundedness_score").alias("avg_groundedness_score"),
        F.avg("relevance_score").alias("avg_relevance_score"),
        F.avg("reference_coverage_score").alias("avg_reference_coverage_score"),
        F.avg(F.col("hallucination_detected").cast("double")).alias("hallucination_rate"),
        F.avg(F.col("major_omission_detected").cast("double")).alias("major_omission_rate"),

        F.avg("retrieval_latency_ms").alias("avg_retrieval_latency_ms"),
        F.expr("percentile_approx(retrieval_latency_ms, 0.95)").alias("p95_retrieval_latency_ms"),
        F.avg("generation_latency_ms").alias("avg_generation_latency_ms"),
        F.expr("percentile_approx(generation_latency_ms, 0.95)").alias("p95_generation_latency_ms"),
        F.avg("total_latency_ms").alias("avg_total_latency_ms"),
        F.expr("percentile_approx(total_latency_ms, 0.95)").alias("p95_total_latency_ms"),
    )
    .first()
    .asDict()
)

summary = {
    "evaluation_version": EVALUATION_VERSION,
    "pipeline_version": PIPELINE_VERSION,
    "mlflow_run_id": evaluation_run_id,
    "benchmark_rows": total_cases,
    "passed_cases": passed_cases,
    "failed_cases": failed_cases,
    "failure_rate": failed_cases / total_cases if total_cases else 0.0,
    "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    **metrics_row,
}

summary_df = spark.createDataFrame([summary])

(
    summary_df.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(SUMMARY_TABLE)
)

display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Log aggregate metrics to the existing MLflow run

# COMMAND ----------

numeric_metrics = {
    key: float(value)
    for key, value in summary.items()
    if isinstance(value, (int, float)) and value is not None
}

with mlflow.start_run(run_id=evaluation_run_id):
    mlflow.log_metrics(numeric_metrics)

    mlflow.log_text(
        json.dumps(summary, indent=2, default=str),
        "rag_evaluation_summary.json",
    )

print("Aggregate evaluation metrics logged to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Failure analysis
# MAGIC
# MAGIC We classify failures without hiding the raw metrics.

# COMMAND ----------

failure_analysis_df = (
    run_df
    .withColumn(
        "failure_category",
        F.when(F.col("status") != "PASSED", F.lit("RUNTIME_FAILURE"))
         .when(~F.col("hit_at_5"), F.lit("RETRIEVAL_FAILURE"))
         .when(F.col("hallucination_detected") == True, F.lit("GROUNDING_FAILURE"))
         .when(F.col("citation_validity_pass") == False, F.lit("CITATION_FAILURE"))
         .when(F.col("gold_evidence_cited") == False, F.lit("GOLD_CITATION_MISS"))
         .when(F.col("relevance_score") < 4.0, F.lit("RELEVANCE_WEAK"))
         .when(F.col("reference_coverage_score") < 4.0, F.lit("INCOMPLETE_ANSWER"))
         .otherwise(F.lit("PASS"))
    )
)

display(
    failure_analysis_df
    .groupBy("failure_category")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Inspect the weakest cases
# MAGIC
# MAGIC These are the cases we should review before changing prompts, retrieval,
# MAGIC reranking, or chunking.

# COMMAND ----------

display(
    failure_analysis_df
    .filter(F.col("failure_category") != "PASS")
    .select(
        "eval_id",
        "question",
        "question_type",
        "difficulty",
        "expected_report_year",
        "failure_category",
        "gold_rank",
        "groundedness_score",
        "relevance_score",
        "reference_coverage_score",
        "citation_validity_pass",
        "gold_evidence_cited",
        "hallucination_detected",
        "major_omission_detected",
        "answer",
        "error_type",
        "error_message",
    )
    .orderBy(
        F.asc_nulls_last("groundedness_score"),
        F.asc_nulls_last("reference_coverage_score"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Slice metrics by question type and difficulty

# COMMAND ----------

display(
    passed_df
    .groupBy("question_type")
    .agg(
        F.count("*").alias("n"),
        F.avg(F.col("hit_at_5").cast("double")).alias("hit_at_5"),
        F.avg("groundedness_score").alias("groundedness"),
        F.avg("relevance_score").alias("relevance"),
        F.avg("reference_coverage_score").alias("reference_coverage"),
        F.avg("total_latency_ms").alias("avg_latency_ms"),
    )
    .orderBy("question_type")
)

display(
    passed_df
    .groupBy("difficulty")
    .agg(
        F.count("*").alias("n"),
        F.avg(F.col("hit_at_5").cast("double")).alias("hit_at_5"),
        F.avg("groundedness_score").alias("groundedness"),
        F.avg("relevance_score").alias("relevance"),
        F.avg("reference_coverage_score").alias("reference_coverage"),
    )
    .orderBy("difficulty")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Evaluation gates
# MAGIC
# MAGIC These are reporting gates, not fabricated claims of production readiness.
# MAGIC We print them so the experiment has explicit quality targets.
# MAGIC Do not tune the system merely to pass a gate without reviewing failure cases.

# COMMAND ----------

QUALITY_GATES = {
    "hit_at_5": 0.90,
    "citation_validity_rate": 0.98,
    "answer_citation_rate": 0.98,
    "avg_groundedness_score": 4.0,
    "avg_relevance_score": 4.0,
    "avg_reference_coverage_score": 4.0,
    "failure_rate": 0.05,  # maximum
}

gate_results = {
    "hit_at_5": summary["hit_at_5"] >= QUALITY_GATES["hit_at_5"],
    "citation_validity_rate": (
        summary["citation_validity_rate"] >= QUALITY_GATES["citation_validity_rate"]
    ),
    "answer_citation_rate": (
        summary["answer_citation_rate"] >= QUALITY_GATES["answer_citation_rate"]
    ),
    "avg_groundedness_score": (
        summary["avg_groundedness_score"] >= QUALITY_GATES["avg_groundedness_score"]
    ),
    "avg_relevance_score": (
        summary["avg_relevance_score"] >= QUALITY_GATES["avg_relevance_score"]
    ),
    "avg_reference_coverage_score": (
        summary["avg_reference_coverage_score"]
        >= QUALITY_GATES["avg_reference_coverage_score"]
    ),
    "failure_rate": summary["failure_rate"] <= QUALITY_GATES["failure_rate"],
}

print("QUALITY GATES")
print("-------------")

for metric, passed in gate_results.items():
    print(
        f"{metric:32s} "
        f"{'PASS' if passed else 'REVIEW'} | "
        f"actual={summary.get(metric)} | "
        f"target={QUALITY_GATES[metric]}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Final validation

# COMMAND ----------

assert spark.catalog.tableExists(RESULTS_TABLE)
assert spark.catalog.tableExists(SUMMARY_TABLE)

persisted_rows = (
    spark.table(RESULTS_TABLE)
    .filter(F.col("mlflow_run_id") == evaluation_run_id)
    .count()
)

assert persisted_rows == len(benchmark_rows)

print("")
print("11_rag_evaluation COMPLETE")
print("")
print("Evaluation run:", evaluation_run_id)
print("Cases:", total_cases)
print("Passed execution:", passed_cases)
print("Failed execution:", failed_cases)
print("")
print("Persisted:")
print(" -", RESULTS_TABLE)
print(" -", SUMMARY_TABLE)
print("")
print("Next decision:")
print(
    "Review failure categories and weakest cases before changing "
    "retrieval, prompts, reranking, or chunking."
)