# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_retrieval_evaluation
# MAGIC
# MAGIC Purpose:
# MAGIC - Evaluate V1/V2/V3 against the approved 50-question benchmark
# MAGIC - Compare ANN, HYBRID, and HYBRID + report-year metadata filtering
# MAGIC - Use strategy-neutral ground truth: document + page overlap
# MAGIC - Compute Hit@1/3/5/10 and MRR@10
# MAGIC - Persist per-query results and experiment summary
# MAGIC - Log aggregate metrics to MLflow
# MAGIC
# MAGIC **You can use Run All after notebook 04 has completed successfully.**
# MAGIC
# MAGIC No LLM generation occurs in this notebook.

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-ai-search mlflow

# COMMAND ----------

# ============================================================
# 01. Imports and configuration
# ============================================================

import time
import math
import mlflow

from pyspark.sql import functions as F
from databricks.ai_search.client import AISearchClient

CATALOG = "worldbank_ai"
RAG_SCHEMA = "rag"

EVAL_DATASET = f"{CATALOG}.{RAG_SCHEMA}.retrieval_eval_dataset"
RESULTS_TABLE = f"{CATALOG}.{RAG_SCHEMA}.retrieval_eval_results"
SUMMARY_TABLE = f"{CATALOG}.{RAG_SCHEMA}.retrieval_eval_summary"

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"

INDEX_NAMES = {
    "fixed_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_fixed_v1_qwen3_index",
    "recursive_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_recursive_v1_qwen3_index",
    "structure_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_structure_v1_qwen3_index",
}

MLFLOW_EXPERIMENT = "/Shared/worldbank_ai_retrieval_experiments"

TOP_K = 10

RETURN_COLUMNS = [
    "chunk_id",
    "document_id",
    "report_year",
    "region",
    "section",
    "page_start",
    "page_end",
    "retrieval_text",
]

client = AISearchClient()
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print("Configuration loaded.")

# COMMAND ----------

# ============================================================
# 02. Validate the approved benchmark
# ============================================================

assert spark.catalog.tableExists(EVAL_DATASET), (
    f"Missing benchmark: {EVAL_DATASET}"
)

eval_df = spark.table(EVAL_DATASET)

assert eval_df.count() == 50, (
    f"Expected 50 benchmark questions, found {eval_df.count()}"
)

assert eval_df.filter(F.col("is_approved") != True).count() == 0
assert eval_df.select("eval_id").distinct().count() == 50
assert eval_df.select("question").distinct().count() == 50

required = [
    "eval_id",
    "question",
    "expected_report_year",
    "relevant_document_id",
    "relevant_page_start",
    "relevant_page_end",
    "reference_evidence",
]

missing = sorted(set(required) - set(eval_df.columns))
assert not missing, f"Benchmark missing columns: {missing}"

print("Benchmark validation passed: 50 approved questions.")

display(
    eval_df.groupBy("expected_report_year")
           .count()
           .orderBy("expected_report_year")
)

# COMMAND ----------

# ============================================================
# 03. Load and validate index handles
# ============================================================

indexes = {}

for strategy, index_name in INDEX_NAMES.items():
    index = client.get_index(
        endpoint_name=AI_SEARCH_ENDPOINT,
        index_name=index_name,
    )

    desc = index.describe()
    state = str(
        desc.get("status", {}).get("detailed_state")
        or desc.get("status", {}).get("state")
        or ""
    ).upper()

    assert state.startswith("ONLINE"), (
        f"{strategy} index is not ONLINE: {state}"
    )

    indexes[strategy] = index
    print(f"{strategy}: ONLINE")

# COMMAND ----------

# ============================================================
# 04. Result parser
# ============================================================

def parse_search_results(response):
    """
    Convert AI Search response into a list of dictionaries.

    AI Search returns:
      manifest.columns -> returned column definitions
      result.data_array -> row arrays with score appended
    """

    manifest_columns = response.get("manifest", {}).get("columns", [])

    names = []
    for col in manifest_columns:
        if isinstance(col, dict):
            names.append(col.get("name"))
        else:
            names.append(str(col))

    rows = response.get("result", {}).get("data_array", [])

    parsed = []

    for row in rows:
        # Usually the score is the final value and is not included
        # in manifest.columns. Handle both forms defensively.
        if len(row) == len(names) + 1:
            values = row[:-1]
            score = row[-1]
        else:
            values = row[:len(names)]
            score = row[len(names)] if len(row) > len(names) else None

        item = dict(zip(names, values))
        item["_score"] = score
        parsed.append(item)

    return parsed

# COMMAND ----------

# ============================================================
# 05. Strategy-neutral relevance rule
# ============================================================
# We do NOT require the V3 relevant_chunk_id for V1/V2.
#
# A retrieved chunk counts as relevant when:
#   1. document_id matches the benchmark source document
#   2. retrieved page range overlaps the benchmark page range
#
# This makes the ground truth comparable across chunking methods.

def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None

def page_overlap(
    retrieved_start,
    retrieved_end,
    expected_start,
    expected_end,
):
    rs = safe_int(retrieved_start)
    re = safe_int(retrieved_end)
    es = safe_int(expected_start)
    ee = safe_int(expected_end)

    if None in (rs, re, es, ee):
        return False

    return max(rs, es) <= min(re, ee)

def is_relevant(hit, expected_document_id, expected_page_start, expected_page_end):
    return (
        str(hit.get("document_id")) == str(expected_document_id)
        and page_overlap(
            hit.get("page_start"),
            hit.get("page_end"),
            expected_page_start,
            expected_page_end,
        )
    )

# COMMAND ----------

# ============================================================
# 06. Search configurations
# ============================================================
# First experiment:
#   A. ANN
#   B. HYBRID
#   C. HYBRID + deterministic report-year filter
#
# The report year is known in our benchmark and represents the
# metadata extraction/filtering stage that the final agent will use.

SEARCH_CONFIGS = [
    {
        "name": "ann",
        "query_type": "ANN",
        "use_year_filter": False,
    },
    {
        "name": "hybrid",
        "query_type": "HYBRID",
        "use_year_filter": False,
    },
    {
        "name": "hybrid_year_filter",
        "query_type": "HYBRID",
        "use_year_filter": True,
    },
]

print(SEARCH_CONFIGS)

# COMMAND ----------

# ============================================================
# 07. Run all retrieval experiments
# ============================================================

eval_rows = [r.asDict(recursive=True) for r in eval_df.collect()]

result_rows = []

for strategy, index in indexes.items():

    for config in SEARCH_CONFIGS:

        print(
            f"\nRunning strategy={strategy}, "
            f"search={config['name']}"
        )

        for i, case in enumerate(eval_rows, start=1):

            filters = None

            if config["use_year_filter"]:
                filters = {
                    "report_year": int(case["expected_report_year"])
                }

            started = time.perf_counter()

            response = index.similarity_search(
                query_text=case["question"],
                columns=RETURN_COLUMNS,
                num_results=TOP_K,
                query_type=config["query_type"],
                filters=filters,
            )

            latency_ms = (
                time.perf_counter() - started
            ) * 1000.0

            hits = parse_search_results(response)

            relevant_ranks = []

            for rank, hit in enumerate(hits, start=1):
                if is_relevant(
                    hit,
                    case["relevant_document_id"],
                    case["relevant_page_start"],
                    case["relevant_page_end"],
                ):
                    relevant_ranks.append(rank)

            first_relevant_rank = (
                min(relevant_ranks)
                if relevant_ranks
                else None
            )

            result_rows.append(
                {
                    "eval_id": case["eval_id"],
                    "question": case["question"],
                    "expected_report_year": int(case["expected_report_year"]),
                    "relevant_document_id": case["relevant_document_id"],
                    "relevant_page_start": int(case["relevant_page_start"]),
                    "relevant_page_end": int(case["relevant_page_end"]),
                    "chunking_strategy": strategy,
                    "search_config": config["name"],
                    "query_type": config["query_type"],
                    "used_year_filter": bool(config["use_year_filter"]),
                    "retrieved_count": len(hits),
                    "first_relevant_rank": first_relevant_rank,
                    "hit_at_1": bool(first_relevant_rank is not None and first_relevant_rank <= 1),
                    "hit_at_3": bool(first_relevant_rank is not None and first_relevant_rank <= 3),
                    "hit_at_5": bool(first_relevant_rank is not None and first_relevant_rank <= 5),
                    "hit_at_10": bool(first_relevant_rank is not None and first_relevant_rank <= 10),
                    "reciprocal_rank": (
                        1.0 / first_relevant_rank
                        if first_relevant_rank is not None
                        else 0.0
                    ),
                    "latency_ms": float(latency_ms),
                }
            )

            if i % 10 == 0:
                print(f"  completed {i}/50")

print(f"\nTotal experiment rows: {len(result_rows)}")

# COMMAND ----------

# ============================================================
# 08. Persist per-question retrieval results
# ============================================================

results_df = spark.createDataFrame(result_rows)

(
    results_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(RESULTS_TABLE)
)

print(f"Saved: {RESULTS_TABLE}")
print(f"Rows: {results_df.count()}")

# COMMAND ----------

# ============================================================
# 09. Aggregate metrics
# ============================================================

summary_df = (
    results_df
    .groupBy(
        "chunking_strategy",
        "search_config",
        "query_type",
        "used_year_filter",
    )
    .agg(
        F.count("*").alias("question_count"),
        F.avg(F.col("hit_at_1").cast("double")).alias("hit_at_1"),
        F.avg(F.col("hit_at_3").cast("double")).alias("hit_at_3"),
        F.avg(F.col("hit_at_5").cast("double")).alias("hit_at_5"),
        F.avg(F.col("hit_at_10").cast("double")).alias("hit_at_10"),
        F.avg("reciprocal_rank").alias("mrr_at_10"),
        F.avg("latency_ms").alias("avg_latency_ms"),
        F.expr("percentile_approx(latency_ms, 0.95)").alias("p95_latency_ms"),
    )
)

(
    summary_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SUMMARY_TABLE)
)

display(
    summary_df.orderBy(
        F.desc("hit_at_10"),
        F.desc("mrr_at_10"),
        F.desc("hit_at_5"),
    )
)

# COMMAND ----------

# ============================================================
# 10. Coverage by report year
# ============================================================

year_summary = (
    results_df
    .groupBy(
        "chunking_strategy",
        "search_config",
        "expected_report_year",
    )
    .agg(
        F.count("*").alias("questions"),
        F.avg(F.col("hit_at_5").cast("double")).alias("hit_at_5"),
        F.avg(F.col("hit_at_10").cast("double")).alias("hit_at_10"),
        F.avg("reciprocal_rank").alias("mrr_at_10"),
    )
    .orderBy(
        "chunking_strategy",
        "search_config",
        "expected_report_year",
    )
)

display(year_summary)

# COMMAND ----------

# ============================================================
# 11. Log each experiment configuration to MLflow
# ============================================================

summary_rows = summary_df.collect()

for row in summary_rows:

    run_name = (
        f"{row['chunking_strategy']}__"
        f"{row['search_config']}__qwen3"
    )

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param(
            "chunking_strategy",
            row["chunking_strategy"]
        )
        mlflow.log_param(
            "search_config",
            row["search_config"]
        )
        mlflow.log_param(
            "query_type",
            row["query_type"]
        )
        mlflow.log_param(
            "year_filter",
            row["used_year_filter"]
        )
        mlflow.log_param(
            "embedding_model",
            "databricks-qwen3-embedding-0-6b"
        )
        mlflow.log_param(
            "benchmark_table",
            EVAL_DATASET
        )
        mlflow.log_param(
            "benchmark_size",
            int(row["question_count"])
        )

        mlflow.log_metric("hit_at_1", float(row["hit_at_1"]))
        mlflow.log_metric("hit_at_3", float(row["hit_at_3"]))
        mlflow.log_metric("hit_at_5", float(row["hit_at_5"]))
        mlflow.log_metric("hit_at_10", float(row["hit_at_10"]))
        mlflow.log_metric("mrr_at_10", float(row["mrr_at_10"]))
        mlflow.log_metric("avg_latency_ms", float(row["avg_latency_ms"]))
        mlflow.log_metric("p95_latency_ms", float(row["p95_latency_ms"]))

print("MLflow logging complete.")

# COMMAND ----------

# ============================================================
# 12. Miss analysis
# ============================================================
# Inspect failures rather than choosing a winner only from one
# aggregate metric.

misses = (
    results_df
    .filter(F.col("hit_at_10") == False)
    .select(
        "chunking_strategy",
        "search_config",
        "eval_id",
        "expected_report_year",
        "question",
        "relevant_document_id",
        "relevant_page_start",
        "relevant_page_end",
        "latency_ms",
    )
    .orderBy(
        "chunking_strategy",
        "search_config",
        "expected_report_year",
        "eval_id",
    )
)

print(f"Top-10 misses: {misses.count()}")
display(misses)

# COMMAND ----------

# ============================================================
# 13. Final experiment table
# ============================================================

final_summary = spark.table(SUMMARY_TABLE)

display(
    final_summary.orderBy(
        F.desc("hit_at_10"),
        F.desc("mrr_at_10"),
        F.desc("hit_at_5"),
    )
)

print("\n05_retrieval_evaluation COMPLETE.")
print(
    "Do NOT choose the production strategy from chunk-size statistics. "
    "Use these measured retrieval results plus miss analysis."
)