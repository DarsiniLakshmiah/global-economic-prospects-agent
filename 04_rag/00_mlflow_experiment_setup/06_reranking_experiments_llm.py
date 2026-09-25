# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 06_reranking_experiments — LLM relevance reranker
# MAGIC
# MAGIC Goal:
# MAGIC - Keep the winning Structure V1 AI Search index.
# MAGIC - Use HYBRID + report-year filtering for candidate generation.
# MAGIC - Retrieve Top 20 candidates.
# MAGIC - Rerank those candidates with the confirmed Databricks-hosted
# MAGIC   `databricks-meta-llama-3-3-70b-instruct` endpoint.
# MAGIC - Compare against the current Top-10 baseline on the SAME 50-question benchmark.
# MAGIC
# MAGIC Important:
# MAGIC - This notebook does NOT rebuild chunks, embeddings, or vector indexes.
# MAGIC - It does NOT use the unavailable Databricks managed reranker.
# MAGIC - The LLM only scores/reorders retrieved evidence. It does not rewrite evidence.

# COMMAND ----------

# Install the Databricks-native OpenAI client as well.
%pip install -q --upgrade databricks-ai-search databricks-openai mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Imports and configuration

# COMMAND ----------

import json
import math
import re
import time
from typing import Any, Dict, List

import mlflow
from openai import OpenAI
from pyspark.sql import functions as F
from databricks.ai_search.client import AISearchClient
from databricks.sdk import WorkspaceClient

CATALOG = "worldbank_ai"
RAG_SCHEMA = "rag"

EVAL_DATASET = f"{CATALOG}.{RAG_SCHEMA}.retrieval_eval_dataset"

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
INDEX_NAME = f"{CATALOG}.{RAG_SCHEMA}.gep_structure_v1_qwen3_index"

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

RESULTS_TABLE = f"{CATALOG}.{RAG_SCHEMA}.reranking_eval_results"
SUMMARY_TABLE = f"{CATALOG}.{RAG_SCHEMA}.reranking_eval_summary"

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_reranking_experiments"

# Baseline returns 10 directly.
BASELINE_TOP_K = 10

# Reranked path retrieves a wider candidate set first.
CANDIDATE_TOP_K = 20
RERANK_TOP_K = 10

# Batch candidates so we do NOT make one LLM call per chunk.
RERANK_BATCH_SIZE = 5

# Keep enough evidence for relevance scoring without sending an
# unnecessarily large prompt.
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
    "retrieval_text",
]

mlflow.set_experiment(MLFLOW_EXPERIMENT)

print("Configuration loaded.")
print("Index:", INDEX_NAME)
print("LLM reranker:", LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Initialize Databricks clients
# MAGIC
# MAGIC Uses notebook/default Databricks authentication.
# MAGIC The notebook-token NOTICE is acceptable for development.

# COMMAND ----------

# Databricks clients
from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI

# AI Search client uses notebook authentication automatically.
ai_search_client = AISearchClient()

index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
)

# Native Databricks OpenAI-compatible client.
# Do NOT manually pass w.config.token here.
llm_client = DatabricksOpenAI()

print("Clients initialized.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Validate benchmark and AI Search index

# COMMAND ----------

assert spark.catalog.tableExists(EVAL_DATASET), f"Missing benchmark table: {EVAL_DATASET}"

eval_df = spark.table(EVAL_DATASET).filter(F.col("is_approved") == True)

eval_count = eval_df.count()

assert eval_count == 50, f"Expected 50 approved questions, found {eval_count}"
assert eval_df.select("eval_id").distinct().count() == 50
assert eval_df.select("question").distinct().count() == 50

index_description = index.describe()
index_status = index_description.get("status", {})

print("Index ready:", index_status.get("ready"))
print("Indexed rows:", index_status.get("indexed_row_count"))
print("Detailed state:", index_status.get("detailed_state"))

assert index_status.get("ready") is True, "Structure V1 AI Search index is not ready."

display(
    eval_df
    .groupBy("expected_report_year")
    .count()
    .orderBy("expected_report_year")
)

print("Benchmark + index validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Verify the LLM endpoint before running the experiment

# COMMAND ----------

test_response = llm_client.chat.completions.create(
    model=LLM_ENDPOINT,
    messages=[
        {
            "role": "user",
            "content": 'Return only this JSON object: {"status":"ok"}'
        }
    ],
    temperature=0,
    max_tokens=30,
)

test_text = test_response.choices[0].message.content

print("LLM endpoint response:", test_text)

assert test_text is not None and len(test_text.strip()) > 0

print("LLM endpoint validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. AI Search response parser

# COMMAND ----------

def _to_dict(obj):
    """Convert Databricks SDK response objects to dictionaries when necessary."""
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "as_dict"):
        return obj.as_dict()

    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    return obj


def parse_search_response(response) -> List[Dict[str, Any]]:
    """Convert AI Search data_array output into row dictionaries."""

    payload = _to_dict(response)

    manifest = _to_dict(payload.get("manifest", {}))
    result = _to_dict(payload.get("result", {}))

    manifest_columns = manifest.get("columns", [])

    column_names = []

    for col in manifest_columns:
        col = _to_dict(col)

        if isinstance(col, dict):
            column_names.append(col.get("name"))
        else:
            column_names.append(getattr(col, "name", None))

    rows = result.get("data_array", []) or []

    return [
        dict(zip(column_names, list(row)))
        for row in rows
    ]


def page_overlap(
    retrieved_start,
    retrieved_end,
    relevant_start,
    relevant_end,
) -> bool:
    """Check whether two page ranges overlap."""

    if None in (
        retrieved_start,
        retrieved_end,
        relevant_start,
        relevant_end,
    ):
        return False

    return (
        int(retrieved_start) <= int(relevant_end)
        and int(retrieved_end) >= int(relevant_start)
    )


def is_relevant(row, benchmark) -> bool:
    """
    Keep the same relevance definition used by Notebook 05:
    correct document + overlapping reference page range.
    """

    return (
        row.get("document_id") == benchmark["relevant_document_id"]
        and page_overlap(
            row.get("page_start"),
            row.get("page_end"),
            benchmark["relevant_page_start"],
            benchmark["relevant_page_end"],
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Hybrid retrieval
# MAGIC
# MAGIC Baseline:
# MAGIC     HYBRID + year filter -> Top 10
# MAGIC
# MAGIC Reranker candidate generation:
# MAGIC     HYBRID + year filter -> Top 20

# COMMAND ----------

def hybrid_search(
    question: str,
    report_year: int,
    num_results: int,
):
    """Run Structure V1 HYBRID search with the strong report-year filter."""

    started = time.perf_counter()

    response = index.similarity_search(
        query_text=question,
        columns=RETURN_COLUMNS,
        num_results=num_results,
        query_type="HYBRID",
        filters={
            "report_year": int(report_year)
        },
    )

    latency_ms = (time.perf_counter() - started) * 1000.0

    rows = parse_search_response(response)

    # Preserve the original retrieval rank for tie-breaking and analysis.
    for rank, row in enumerate(rows, start=1):
        row["_retrieval_rank"] = rank

    return rows, latency_ms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. LLM relevance reranker
# MAGIC
# MAGIC The LLM receives the question plus candidate evidence and returns:
# MAGIC
# MAGIC `chunk_id -> relevance score`
# MAGIC
# MAGIC Score meaning:
# MAGIC - 0 = irrelevant
# MAGIC - 1 = weakly related
# MAGIC - 2 = partially useful
# MAGIC - 3 = relevant
# MAGIC - 4 = highly relevant / directly answers the question
# MAGIC
# MAGIC We use temperature 0 and validate every returned chunk ID.

# COMMAND ----------

SYSTEM_PROMPT = """
You are a retrieval relevance reranker for World Bank Global Economic Prospects reports.

Your only job is to score how useful each candidate passage is for answering the user's question.

Scoring:
0 = irrelevant
1 = weakly related
2 = partially useful
3 = relevant
4 = highly relevant and directly useful

Important rules:
- Judge relevance to the QUESTION, not writing quality.
- Prefer passages containing the evidence needed to answer the question.
- Do not answer the question.
- Do not add facts.
- Do not change candidate text.
- Score every supplied candidate exactly once.
- Use only chunk IDs that were supplied.
- Return JSON only.

Required JSON format:
{
  "scores": [
    {"chunk_id": "exact supplied id", "score": 0}
  ]
}
""".strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parse JSON returned by the LLM.
    Handles an occasional markdown code fence without silently accepting
    malformed/non-JSON output.
    """

    if text is None:
        raise ValueError("LLM returned no content.")

    cleaned = text.strip()

    # Remove markdown fences if the model ignored the JSON-only instruction.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        # Last controlled attempt: isolate the outer JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise

        return json.loads(cleaned[start:end + 1])


def rerank_batch(
    question: str,
    candidates: List[Dict[str, Any]],
    max_retries: int = 2,
):
    """
    Score one candidate batch using the Databricks-hosted LLM.

    Reliability behavior:
    - Validate JSON output.
    - Validate chunk IDs.
    - Validate scores are 0-4.
    - Retry malformed JSON responses.
    - Retry only missing candidate IDs when possible.
    - Never invent relevance scores.
    """

    # ---------------------------------------
    # Candidates that MUST receive a score
    # ---------------------------------------
    expected_ids = {
        str(row["chunk_id"])
        for row in candidates
    }

    candidate_lookup = {
        str(row["chunk_id"]): row
        for row in candidates
    }

    score_map = {}

    total_latency_ms = 0.0
    total_llm_calls = 0

    # Initially score the entire batch.
    pending_ids = set(expected_ids)

    for attempt in range(max_retries + 1):

        if not pending_ids:
            break

        # Only send candidates that still need scores.
        pending_candidates = [
            candidate_lookup[chunk_id]
            for chunk_id in sorted(pending_ids)
        ]

        candidate_payload = []

        for row in pending_candidates:

            candidate_payload.append({
                "chunk_id": str(row["chunk_id"]),
                "report_year": row.get("report_year"),
                "region": row.get("region"),
                "section": row.get("section"),
                "pages": [
                    row.get("page_start"),
                    row.get("page_end"),
                ],
                "text": (
                    row.get("retrieval_text") or ""
                )[:MAX_TEXT_CHARS_PER_CANDIDATE],
            })

        # ---------------------------------------
        # Stronger JSON-only instruction
        # ---------------------------------------
        user_prompt = f"""
QUESTION:
{question}

Score EVERY candidate below exactly once.

Return ONLY valid JSON.

Required format:

{{
  "scores": [
    {{
      "chunk_id": "exact_chunk_id",
      "score": 0
    }}
  ]
}}

Rules:
- Return one entry for EVERY supplied chunk_id.
- Use the exact chunk_id.
- score must be a number from 0 to 4.
- Do not include explanations.
- Do not include markdown.
- Do not include comments.
- Do not include trailing commas.
- Do not omit candidates.

CANDIDATES:

{json.dumps(candidate_payload, ensure_ascii=False)}
""".strip()

        # ---------------------------------------
        # LLM call
        # ---------------------------------------
        started = time.perf_counter()

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
            temperature=0,
            max_tokens=1200,
        )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        total_latency_ms += latency_ms
        total_llm_calls += 1

        raw_text = response.choices[0].message.content

        # ---------------------------------------
        # Parse JSON safely
        # ---------------------------------------
        try:

            parsed = extract_json_object(raw_text)

        except (json.JSONDecodeError, ValueError) as e:

            print(
                f"Malformed reranker JSON. "
                f"Retry {attempt + 1}/{max_retries}. "
                f"Error: {str(e)}"
            )

            # Do NOT change pending_ids.
            # Retry the same candidates.
            continue

        scores = parsed.get("scores")

        if not isinstance(scores, list):

            print(
                f"Reranker returned JSON without a valid "
                f"'scores' list. "
                f"Retry {attempt + 1}/{max_retries}."
            )

            continue

        # ---------------------------------------
        # Validate returned scores
        # ---------------------------------------
        response_had_valid_score = False

        for item in scores:

            if not isinstance(item, dict):
                continue

            chunk_id = item.get("chunk_id")

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id)

            # Prevent hallucinated IDs.
            if chunk_id not in expected_ids:

                print(
                    "Ignoring unknown chunk_id returned "
                    f"by reranker: {chunk_id}"
                )

                continue

            # Already successfully scored.
            if chunk_id in score_map:
                continue

            score = item.get("score")

            try:
                score = float(score)

            except (TypeError, ValueError):

                print(
                    f"Ignoring invalid score for "
                    f"{chunk_id}: {score}"
                )

                continue

            if not 0 <= score <= 4:

                print(
                    f"Ignoring out-of-range score for "
                    f"{chunk_id}: {score}"
                )

                continue

            score_map[chunk_id] = score

            response_had_valid_score = True

        # ---------------------------------------
        # Determine what still needs scoring
        # ---------------------------------------
        pending_ids = (
            expected_ids - set(score_map.keys())
        )

        if pending_ids:

            print(
                f"Reranker still needs "
                f"{len(pending_ids)} candidate(s). "
                f"Retry {attempt + 1}/{max_retries}: "
                f"{sorted(pending_ids)}"
            )

    # ---------------------------------------
    # Final strict validation
    # ---------------------------------------
    missing_ids = (
        expected_ids - set(score_map.keys())
    )

    if missing_ids:

        raise ValueError(
            "LLM reranker could not produce valid scores "
            "after retries for candidate IDs: "
            f"{sorted(missing_ids)}"
        )

    return (
        score_map,
        total_latency_ms,
        total_llm_calls,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Full reranking function
# MAGIC
# MAGIC Candidates are scored in batches of 10.
# MAGIC
# MAGIC With Top 20 retrieval:
# MAGIC - 2 LLM calls per question
# MAGIC - 100 LLM reranking calls for the 50-question benchmark
# MAGIC
# MAGIC Original retrieval rank is used only as a deterministic tie-breaker.

# COMMAND ----------

def llm_rerank(
    question: str,
    candidates: List[Dict[str, Any]],
):
    """
    Rerank all retrieved candidates using batched LLM relevance scoring.

    Normal case:
        20 candidates
        -> 2 batches of 10
        -> 2 LLM calls

    If a candidate is omitted:
        -> retry only missing candidate(s)
        -> count the additional call
    """

    score_map = {}

    total_llm_latency_ms = 0.0
    total_llm_calls = 0

    # Process candidates in batches.
    for start in range(
        0,
        len(candidates),
        RERANK_BATCH_SIZE,
    ):

        batch = candidates[
            start:start + RERANK_BATCH_SIZE
        ]

        # rerank_batch now returns:
        # 1. relevance scores
        # 2. total latency including retries
        # 3. actual number of LLM calls including retries
        (
            batch_scores,
            batch_latency_ms,
            batch_llm_calls,
        ) = rerank_batch(
            question=question,
            candidates=batch,
        )

        score_map.update(batch_scores)

        total_llm_latency_ms += (
            batch_latency_ms
        )

        total_llm_calls += (
            batch_llm_calls
        )

    # ----------------------------------
    # Attach scores to original evidence
    # ----------------------------------
    reranked = []

    for row in candidates:

        enriched = dict(row)

        enriched["_rerank_score"] = (
            score_map[str(row["chunk_id"])]
        )

        reranked.append(enriched)

    # ----------------------------------
    # Sort by LLM relevance
    # ----------------------------------
    # Highest relevance score first.
    #
    # If two chunks receive the same LLM score,
    # preserve the better original retrieval rank.
    reranked.sort(
        key=lambda row: (
            -row["_rerank_score"],
            row["_retrieval_rank"],
        )
    )

    # Keep only final Top 10.
    return (
        reranked[:RERANK_TOP_K],
        total_llm_latency_ms,
        total_llm_calls,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Smoke test
# MAGIC
# MAGIC Run one real benchmark question through:
# MAGIC
# MAGIC 1. baseline retrieval
# MAGIC 2. Top-20 candidate retrieval
# MAGIC 3. LLM reranking
# MAGIC
# MAGIC If this cell passes, the full experiment can proceed.

# COMMAND ----------

sample = eval_df.orderBy("eval_id").first().asDict(recursive=True)

print("Question:", sample["question"])
print("Expected year:", sample["expected_report_year"])

baseline_rows, baseline_retrieval_ms = hybrid_search(
    question=sample["question"],
    report_year=sample["expected_report_year"],
    num_results=BASELINE_TOP_K,
)

candidate_rows, candidate_retrieval_ms = hybrid_search(
    question=sample["question"],
    report_year=sample["expected_report_year"],
    num_results=CANDIDATE_TOP_K,
)

reranked_rows, rerank_llm_ms, rerank_calls = llm_rerank(
    question=sample["question"],
    candidates=candidate_rows,
)

assert len(baseline_rows) > 0
assert len(candidate_rows) > 0
assert len(reranked_rows) > 0

print("\nBaseline results:", len(baseline_rows))
print("Candidate results:", len(candidate_rows))
print("Reranked results:", len(reranked_rows))

print("\nBaseline retrieval latency ms:", round(baseline_retrieval_ms, 2))
print("Candidate retrieval latency ms:", round(candidate_retrieval_ms, 2))
print("LLM reranking latency ms:", round(rerank_llm_ms, 2))
print("LLM calls:", rerank_calls)

print("\nTop reranked candidates:")

for rank, row in enumerate(reranked_rows[:5], start=1):
    print(
        rank,
        "| score =", row["_rerank_score"],
        "| original rank =", row["_retrieval_rank"],
        "| pages =", row.get("page_start"), "-", row.get("page_end"),
        "| chunk =", row.get("chunk_id"),
    )

print("\nSmoke test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Evaluation metrics

# COMMAND ----------

def evaluate_ranked_results(
    rows: List[Dict[str, Any]],
    benchmark: Dict[str, Any],
):
    """Compute retrieval metrics from the ordered result list."""

    relevant_ranks = [
        rank
        for rank, row in enumerate(rows, start=1)
        if is_relevant(row, benchmark)
    ]

    first_rank = min(relevant_ranks) if relevant_ranks else None

    return {
        "first_relevant_rank": first_rank,
        "hit_at_1": int(first_rank is not None and first_rank <= 1),
        "hit_at_3": int(first_rank is not None and first_rank <= 3),
        "hit_at_5": int(first_rank is not None and first_rank <= 5),
        "hit_at_10": int(first_rank is not None and first_rank <= 10),
        "mrr_at_10": (
            1.0 / first_rank
            if first_rank is not None and first_rank <= 10
            else 0.0
        ),
    }


def percentile(values, p):
    """Small deterministic percentile helper."""

    values = sorted(float(v) for v in values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    return (
        values[lower]
        + (values[upper] - values[lower])
        * (position - lower)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Run the full 50-question comparison
# MAGIC
# MAGIC This is the longest cell.
# MAGIC
# MAGIC Per question:
# MAGIC - one Top-20 HYBRID retrieval
# MAGIC - baseline = first 10 results from that same retrieval
# MAGIC - two batched LLM reranker calls
# MAGIC
# MAGIC Reusing the Top-20 retrieval for both paths makes the comparison cleaner
# MAGIC and avoids an unnecessary second search request.

# COMMAND ----------

benchmark_rows = [
    row.asDict(recursive=True)
    for row in eval_df.orderBy("eval_id").collect()
]

records = []

for i, benchmark in enumerate(benchmark_rows, start=1):

    question = benchmark["question"]
    report_year = benchmark["expected_report_year"]

    # One shared candidate retrieval.
    candidates, retrieval_latency_ms = hybrid_search(
        question=question,
        report_year=report_year,
        num_results=CANDIDATE_TOP_K,
    )

    assert len(candidates) > 0, (
        f"No candidates returned for eval_id={benchmark['eval_id']}"
    )

    # Baseline is exactly the first 10 results from the same ranked retrieval.
    baseline_rows = candidates[:BASELINE_TOP_K]

    baseline_metrics = evaluate_ranked_results(
        baseline_rows,
        benchmark,
    )

    records.append({
        "eval_id": benchmark["eval_id"],
        "question": question,
        "expected_report_year": int(report_year),
        "relevant_document_id": benchmark["relevant_document_id"],
        "relevant_page_start": int(benchmark["relevant_page_start"]),
        "relevant_page_end": int(benchmark["relevant_page_end"]),
        "chunking_strategy": "structure_v1",
        "retrieval_config": "hybrid_year_filter",
        "first_relevant_rank": baseline_metrics["first_relevant_rank"],
        "hit_at_1": baseline_metrics["hit_at_1"],
        "hit_at_3": baseline_metrics["hit_at_3"],
        "hit_at_5": baseline_metrics["hit_at_5"],
        "hit_at_10": baseline_metrics["hit_at_10"],
        "mrr_at_10": baseline_metrics["mrr_at_10"],
        "retrieval_latency_ms": float(retrieval_latency_ms),
        "rerank_latency_ms": 0.0,
        "total_latency_ms": float(retrieval_latency_ms),
        "llm_rerank_calls": 0,
        "returned_chunk_ids_json": json.dumps(
            [row.get("chunk_id") for row in baseline_rows]
        ),
        "rerank_scores_json": None,
    })

    # LLM reranking.
    reranked_rows, rerank_latency_ms, llm_calls = llm_rerank(
        question=question,
        candidates=candidates,
    )

    reranked_metrics = evaluate_ranked_results(
        reranked_rows,
        benchmark,
    )

    records.append({
        "eval_id": benchmark["eval_id"],
        "question": question,
        "expected_report_year": int(report_year),
        "relevant_document_id": benchmark["relevant_document_id"],
        "relevant_page_start": int(benchmark["relevant_page_start"]),
        "relevant_page_end": int(benchmark["relevant_page_end"]),
        "chunking_strategy": "structure_v1",
        "retrieval_config": "hybrid_year_filter_llm_reranked",
        "first_relevant_rank": reranked_metrics["first_relevant_rank"],
        "hit_at_1": reranked_metrics["hit_at_1"],
        "hit_at_3": reranked_metrics["hit_at_3"],
        "hit_at_5": reranked_metrics["hit_at_5"],
        "hit_at_10": reranked_metrics["hit_at_10"],
        "mrr_at_10": reranked_metrics["mrr_at_10"],
        "retrieval_latency_ms": float(retrieval_latency_ms),
        "rerank_latency_ms": float(rerank_latency_ms),
        "total_latency_ms": float(
            retrieval_latency_ms + rerank_latency_ms
        ),
        "llm_rerank_calls": int(llm_calls),
        "returned_chunk_ids_json": json.dumps(
            [row.get("chunk_id") for row in reranked_rows]
        ),
        "rerank_scores_json": json.dumps({
            str(row.get("chunk_id")): row.get("_rerank_score")
            for row in reranked_rows
        }),
    })

    if i % 5 == 0 or i == len(benchmark_rows):
        print(f"Completed {i}/{len(benchmark_rows)} questions")

assert len(records) == 100, (
    f"Expected 100 comparison records, found {len(records)}"
)

print("\nFull reranking experiment complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Persist per-query results

# COMMAND ----------

results_df = spark.createDataFrame(records)

(
    results_df
    .withColumn("evaluated_at", F.current_timestamp())
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(RESULTS_TABLE)
)

saved_results_count = spark.table(RESULTS_TABLE).count()

assert saved_results_count == 100

print("Saved:", RESULTS_TABLE)
print("Rows:", saved_results_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Aggregate comparison

# COMMAND ----------

summary_rows = []

for config_name in [
    "hybrid_year_filter",
    "hybrid_year_filter_llm_reranked",
]:

    subset = [
        row
        for row in records
        if row["retrieval_config"] == config_name
    ]

    total_latencies = [
        row["total_latency_ms"]
        for row in subset
    ]

    rerank_latencies = [
        row["rerank_latency_ms"]
        for row in subset
    ]

    summary_rows.append({
        "chunking_strategy": "structure_v1",
        "retrieval_config": config_name,
        "question_count": len(subset),
        "hit_at_1": sum(row["hit_at_1"] for row in subset) / len(subset),
        "hit_at_3": sum(row["hit_at_3"] for row in subset) / len(subset),
        "hit_at_5": sum(row["hit_at_5"] for row in subset) / len(subset),
        "hit_at_10": sum(row["hit_at_10"] for row in subset) / len(subset),
        "mrr_at_10": sum(row["mrr_at_10"] for row in subset) / len(subset),
        "avg_total_latency_ms": sum(total_latencies) / len(total_latencies),
        "p95_total_latency_ms": percentile(total_latencies, 0.95),
        "avg_rerank_latency_ms": sum(rerank_latencies) / len(rerank_latencies),
        "total_llm_rerank_calls": sum(
            row["llm_rerank_calls"]
            for row in subset
        ),
    })

summary_df = spark.createDataFrame(summary_rows)

(
    summary_df
    .withColumn("evaluated_at", F.current_timestamp())
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SUMMARY_TABLE)
)

display(
    summary_df.orderBy(
        F.desc("mrr_at_10"),
        F.desc("hit_at_5"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Measure where reranking helped or hurt

# COMMAND ----------

baseline = (
    spark.table(RESULTS_TABLE)
    .filter(F.col("retrieval_config") == "hybrid_year_filter")
    .select(
        "eval_id",
        "question",
        F.col("first_relevant_rank").alias("baseline_rank"),
        F.col("hit_at_5").alias("baseline_hit5"),
        F.col("mrr_at_10").alias("baseline_mrr"),
    )
)

reranked = (
    spark.table(RESULTS_TABLE)
    .filter(
        F.col("retrieval_config")
        == "hybrid_year_filter_llm_reranked"
    )
    .select(
        "eval_id",
        F.col("first_relevant_rank").alias("reranked_rank"),
        F.col("hit_at_5").alias("reranked_hit5"),
        F.col("mrr_at_10").alias("reranked_mrr"),
        "rerank_latency_ms",
    )
)

comparison_df = (
    baseline
    .join(reranked, on="eval_id", how="inner")
    .withColumn(
        "mrr_delta",
        F.col("reranked_mrr") - F.col("baseline_mrr"),
    )
    .withColumn(
        "hit5_delta",
        F.col("reranked_hit5") - F.col("baseline_hit5"),
    )
)

display(
    comparison_df.orderBy(
        F.desc("mrr_delta"),
        "eval_id",
    )
)

print(
    "Questions with improved MRR:",
    comparison_df.filter(F.col("mrr_delta") > 0).count(),
)

print(
    "Questions with worse MRR:",
    comparison_df.filter(F.col("mrr_delta") < 0).count(),
)

print(
    "Questions unchanged:",
    comparison_df.filter(F.col("mrr_delta") == 0).count(),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Inspect remaining Hit@5 misses after reranking

# COMMAND ----------

reranked_misses = (
    spark.table(RESULTS_TABLE)
    .filter(
        (F.col("retrieval_config") == "hybrid_year_filter_llm_reranked")
        & (F.col("hit_at_5") == 0)
    )
    .select(
        "eval_id",
        "question",
        "expected_report_year",
        "relevant_document_id",
        "relevant_page_start",
        "relevant_page_end",
        "first_relevant_rank",
        "rerank_latency_ms",
        "returned_chunk_ids_json",
        "rerank_scores_json",
    )
    .orderBy("eval_id")
)

print("Reranked Hit@5 misses:", reranked_misses.count())

display(reranked_misses)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Log experiment to MLflow

# COMMAND ----------

for summary in summary_rows:

    with mlflow.start_run(
        run_name=f"structure_v1__{summary['retrieval_config']}"
    ):

        mlflow.log_params({
            "chunking_strategy": "structure_v1",
            "retrieval_config": summary["retrieval_config"],
            "index_name": INDEX_NAME,
            "embedding_endpoint": "databricks-qwen3-embedding-0-6b",
            "reranker_endpoint": (
                LLM_ENDPOINT
                if "llm_reranked" in summary["retrieval_config"]
                else "none"
            ),
            "benchmark_table": EVAL_DATASET,
            "question_count": summary["question_count"],
            "candidate_top_k": (
                CANDIDATE_TOP_K
                if "llm_reranked" in summary["retrieval_config"]
                else BASELINE_TOP_K
            ),
            "final_top_k": 10,
            "query_type": "HYBRID",
            "report_year_filter": True,
            "rerank_batch_size": RERANK_BATCH_SIZE,
            "temperature": 0,
        })

        mlflow.log_metrics({
            "hit_at_1": summary["hit_at_1"],
            "hit_at_3": summary["hit_at_3"],
            "hit_at_5": summary["hit_at_5"],
            "hit_at_10": summary["hit_at_10"],
            "mrr_at_10": summary["mrr_at_10"],
            "avg_total_latency_ms": summary["avg_total_latency_ms"],
            "p95_total_latency_ms": summary["p95_total_latency_ms"],
            "avg_rerank_latency_ms": summary["avg_rerank_latency_ms"],
            "total_llm_rerank_calls": summary["total_llm_rerank_calls"],
        })

print("MLflow runs logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Final validation

# COMMAND ----------

assert spark.table(RESULTS_TABLE).count() == 100
assert spark.table(SUMMARY_TABLE).count() == 2

configs_found = {
    row["retrieval_config"]
    for row in spark.table(SUMMARY_TABLE)
        .select("retrieval_config")
        .collect()
}

assert configs_found == {
    "hybrid_year_filter",
    "hybrid_year_filter_llm_reranked",
}

print("06_reranking_experiments COMPLETE")
print("")
print("Results table:", RESULTS_TABLE)
print("Summary table:", SUMMARY_TABLE)
print("")
print("Decision rule:")
print("- Keep reranking only if relevance improves enough to justify its latency.")
print("- Do not assume the LLM reranker is better until the measured results say so.")