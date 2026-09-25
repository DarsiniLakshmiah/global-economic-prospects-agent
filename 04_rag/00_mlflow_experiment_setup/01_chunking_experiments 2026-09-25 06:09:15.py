# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 - Chunking Experiments
# MAGIC Compare three deterministic chunking candidates for the World Bank Global Economic Prospects corpus.
# MAGIC
# MAGIC - V1: fixed-size chunks
# MAGIC - V2: recursive / paragraph-aware chunks with overlap
# MAGIC - V3: existing structure-aware parent-child baseline
# MAGIC
# MAGIC This notebook profiles and logs structural metrics. It does **not** select a winner; retrieval quality is evaluated later.

# COMMAND ----------

# Configuration
CATALOG = "worldbank_ai"
SILVER_SCHEMA = "silver"
RAG_SCHEMA = "rag"

CLEAN_PAGE_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.gep_clean_pages"
BASELINE_CHILD_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.gep_child_chunks_enriched"

FIXED_TABLE = f"{CATALOG}.{RAG_SCHEMA}.gep_chunks_fixed_v1"
RECURSIVE_TABLE = f"{CATALOG}.{RAG_SCHEMA}.gep_chunks_recursive_v1"
STRUCTURE_TABLE = f"{CATALOG}.{RAG_SCHEMA}.gep_chunks_structure_v1"
SUMMARY_TABLE = f"{CATALOG}.{RAG_SCHEMA}.chunking_experiment_summary"

# ~4 chars/token is only an operational approximation.
TARGET_CHARS = 2000
MAX_CHARS = 2500
OVERLAP_CHARS = 300
MIN_QUALITY_CHARS = 300

print("Configuration loaded.")
print(f"Clean source:      {CLEAN_PAGE_TABLE}")
print(f"Structure source:  {BASELINE_CHILD_TABLE}")

# COMMAND ----------

# Imports
import re
import math
import hashlib
import json
import mlflow

from pyspark.sql import functions as F
from pyspark.sql import types as T

print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------

# Ensure RAG schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{RAG_SCHEMA}")
print(f"RAG schema ready: {CATALOG}.{RAG_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow experiment
# MAGIC If `00_mlflow_experiment_setup` created a different experiment path, change only `EXPERIMENT_NAME` below.

# COMMAND ----------

# ============================================================
# Configure MLflow experiment
# ============================================================
#
# Store the experiment directly under /Shared.
# This avoids requiring a separate /Shared/worldbank_ai
# workspace directory.
# ============================================================

import mlflow

EXPERIMENT_NAME = "/Shared/worldbank_ai_chunking_experiments"

# Creates the experiment automatically if it does not exist.
mlflow.set_experiment(EXPERIMENT_NAME)

# Read it back to verify creation.
experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise RuntimeError(
        f"MLflow experiment was not created: "
        f"{EXPERIMENT_NAME}"
    )

print("MLflow experiment configured.")
print(f"Name: {EXPERIMENT_NAME}")
print(f"ID:   {experiment.experiment_id}")

# COMMAND ----------

# Load source datasets
clean_pages_df = spark.table(CLEAN_PAGE_TABLE)
structure_baseline_df = spark.table(BASELINE_CHILD_TABLE)

clean_page_count = clean_pages_df.count()
structure_chunk_count = structure_baseline_df.count()

print(f"Clean pages: {clean_page_count:,}")
print(f"Structure-aware baseline chunks: {structure_chunk_count:,}")

if clean_page_count != 1098:
    raise RuntimeError(f"Expected 1,098 clean pages, found {clean_page_count:,}.")

expected_years = {2022, 2023, 2024, 2025, 2026}
actual_years = {
    row["report_year"]
    for row in clean_pages_df.select("report_year").distinct().collect()
}
if actual_years != expected_years:
    raise RuntimeError(f"Unexpected report years: {actual_years}")

print("Input validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect source schema
# MAGIC This cell makes schema mismatches obvious before chunk generation. The next cell resolves common column-name variants without modifying source data.

# COMMAND ----------

print("Clean-page columns:")
print(clean_pages_df.columns)

print("\nStructure baseline columns:")
print(structure_baseline_df.columns)

# COMMAND ----------

# Resolve common source-column variants from the clean-page table.
def first_existing(columns, candidates, required=True):
    for name in candidates:
        if name in columns:
            return name
    if required:
        raise RuntimeError(
            f"None of the expected columns exist. Candidates: {candidates}. "
            f"Available: {columns}"
        )
    return None

clean_cols = clean_pages_df.columns

text_col = first_existing(clean_cols, ["clean_text", "cleaned_text", "page_text", "text"])
page_col = first_existing(clean_cols, ["page_number", "page_num", "page"])
filename_col = first_existing(clean_cols, ["filename", "source_filename", "file_name"], required=False)
filepath_col = first_existing(clean_cols, ["file_path", "source_file_path", "source_path"], required=False)

select_exprs = [
    F.col("document_id"),
    F.col("report_year").cast("int").alias("report_year"),
    F.col(page_col).cast("int").alias("page_number"),
    F.col(text_col).alias("clean_text"),
]
if filename_col:
    select_exprs.append(F.col(filename_col).alias("filename"))
else:
    select_exprs.append(F.lit(None).cast("string").alias("filename"))
if filepath_col:
    select_exprs.append(F.col(filepath_col).alias("file_path"))
else:
    select_exprs.append(F.lit(None).cast("string").alias("file_path"))

source_pages_df = (
    clean_pages_df
    .filter(F.col(text_col).isNotNull())
    .filter(F.length(F.trim(F.col(text_col))) > 0)
    .select(*select_exprs)
    .orderBy("document_id", "page_number")
)

usable_page_count = source_pages_df.count()
print(f"Resolved text column: {text_col}")
print(f"Resolved page column: {page_col}")
print(f"Usable pages for chunking: {usable_page_count:,}")

# COMMAND ----------

# Stable chunk ID helper
def make_chunk_id(method: str, document_id: str, page_number: int, chunk_index: int, text: str) -> str:
    """Create a deterministic, reproducible chunk identifier."""
    normalized_text = text.strip().encode("utf-8", errors="ignore")
    text_hash = hashlib.sha1(normalized_text).hexdigest()[:12]
    return f"{method}__{document_id}__p{page_number:04d}__c{chunk_index:04d}__{text_hash}"

# COMMAND ----------

# V1 - Fixed-size chunking
def fixed_size_chunks(text: str, target_chars: int = TARGET_CHARS):
    """Split text into non-overlapping fixed-size character windows."""
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

# Sanity test
sample_chunks = fixed_size_chunks("A" * 4500, target_chars=2000)
assert [len(x) for x in sample_chunks] == [2000, 2000, 500]
print("Fixed-size chunking test passed.")

# COMMAND ----------

# V2 helpers - recursive / paragraph-aware chunking
def split_sentences(text: str):
    """Lightweight deterministic sentence splitting."""
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def hard_split_text(text: str, max_chars: int):
    """Last-resort split, preferring whitespace near max_chars."""
    pieces = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < int(max_chars * 0.60):
            split_at = max_chars
        piece = remaining[:split_at].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def recursive_blocks(text: str, max_chars: int = MAX_CHARS):
    """Split by paragraph, then sentence, then word-safe hard split."""
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n+", text)
    output = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            output.append(paragraph)
            continue

        sentences = split_sentences(paragraph)
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    output.append(current.strip())
                    current = ""
                output.extend(hard_split_text(sentence, max_chars))
                continue

            candidate = sentence if not current else current + " " + sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    output.append(current.strip())
                current = sentence

        if current:
            output.append(current.strip())

    return output


def recursive_chunks(text: str, target_chars: int = TARGET_CHARS,
                     max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS):
    """Create paragraph-aware chunks with limited tail overlap."""
    blocks = recursive_blocks(text, max_chars=max_chars)
    if not blocks:
        return []

    chunks = []
    current = ""

    for block in blocks:
        candidate = block if not current else current + "\n\n" + block
        if len(candidate) <= target_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap = current[-overlap_chars:].strip()
            candidate_with_overlap = overlap + "\n\n" + block if overlap else block
            current = candidate_with_overlap if len(candidate_with_overlap) <= max_chars else block
        else:
            current = block

    if current:
        chunks.append(current.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(hard_split_text(chunk, max_chars))
    return final_chunks

# Sanity test
sample_text = """
Global growth weakened during the year as financial conditions tightened across major economies.

Growth in emerging markets remained uneven. Several economies experienced weaker external demand.

Risks to the outlook remain tilted to the downside. Persistent inflation could weaken activity further.
""" * 15

test_chunks = recursive_chunks(sample_text)
assert all(len(chunk) <= MAX_CHARS for chunk in test_chunks)
print(f"Recursive chunking test passed. Generated {len(test_chunks)} chunks.")

# COMMAND ----------

# Collect source pages for this controlled experiment.
# Five reports / ~1,098 pages is small enough for driver-side deterministic generation.
# For a much larger corpus, distribute this logic instead.
source_pages = source_pages_df.orderBy("document_id", "page_number").collect()
print(f"Collected source pages: {len(source_pages):,}")

# COMMAND ----------

# Experimental chunk schema
experiment_chunk_schema = T.StructType([
    T.StructField("chunk_id", T.StringType(), False),
    T.StructField("chunking_method", T.StringType(), False),
    T.StructField("chunking_version", T.StringType(), False),
    T.StructField("document_id", T.StringType(), False),
    T.StructField("report_year", T.IntegerType(), False),
    T.StructField("page_start", T.IntegerType(), False),
    T.StructField("page_end", T.IntegerType(), False),
    T.StructField("chunk_index", T.IntegerType(), False),
    T.StructField("chunk_text", T.StringType(), False),
    T.StructField("character_count", T.IntegerType(), False),
    T.StructField("approx_token_count", T.IntegerType(), False),
])

# COMMAND ----------

# Generate V1 fixed-size chunks
fixed_rows = []
for page in source_pages:
    chunks = fixed_size_chunks(page["clean_text"], target_chars=TARGET_CHARS)
    for index, chunk in enumerate(chunks):
        fixed_rows.append((
            make_chunk_id("fixed_size", page["document_id"], page["page_number"], index, chunk),
            "fixed_size", "v1", page["document_id"], int(page["report_year"]),
            int(page["page_number"]), int(page["page_number"]), int(index), chunk,
            int(len(chunk)), int(math.ceil(len(chunk) / 4.0)),
        ))

fixed_df = spark.createDataFrame(fixed_rows, schema=experiment_chunk_schema)
print(f"V1 fixed chunks: {fixed_df.count():,}")

# COMMAND ----------

# Generate V2 recursive + overlap chunks
recursive_rows = []
for page in source_pages:
    chunks = recursive_chunks(
        page["clean_text"],
        target_chars=TARGET_CHARS,
        max_chars=MAX_CHARS,
        overlap_chars=OVERLAP_CHARS,
    )
    for index, chunk in enumerate(chunks):
        recursive_rows.append((
            make_chunk_id("recursive_overlap", page["document_id"], page["page_number"], index, chunk),
            "recursive_overlap", "v1", page["document_id"], int(page["report_year"]),
            int(page["page_number"]), int(page["page_number"]), int(index), chunk,
            int(len(chunk)), int(math.ceil(len(chunk) / 4.0)),
        ))

recursive_df = spark.createDataFrame(recursive_rows, schema=experiment_chunk_schema)
print(f"V2 recursive chunks: {recursive_df.count():,}")

# COMMAND ----------

# Prepare V3 existing structure-aware baseline.
# Reuse the persisted baseline rather than reimplementing the parser here.
structure_df = (
    structure_baseline_df
    .select(
        "chunk_id",
        F.lit("structure_aware_parent_child").alias("chunking_method"),
        F.lit("baseline_v1").alias("chunking_version"),
        "document_id",
        F.col("report_year").cast("int").alias("report_year"),
        F.col("page_start").cast("int").alias("page_start"),
        F.col("page_end").cast("int").alias("page_end"),
        F.col("chunk_index").cast("int").alias("chunk_index"),
        "chunk_text",
        F.length("chunk_text").cast("int").alias("character_count"),
        F.ceil(F.length("chunk_text") / F.lit(4.0)).cast("int").alias("approx_token_count"),
    )
)
print(f"V3 structure-aware chunks: {structure_df.count():,}")

# COMMAND ----------

# Generic validation shared by all variants
def validate_chunk_dataframe(df, name: str):
    total = df.count()
    if total == 0:
        raise RuntimeError(f"{name} generated zero chunks.")

    null_ids = df.filter(F.col("chunk_id").isNull()).count()
    empty_text = df.filter(
        F.col("chunk_text").isNull() | (F.length(F.trim(F.col("chunk_text"))) == 0)
    ).count()
    duplicate_ids = df.groupBy("chunk_id").count().filter(F.col("count") > 1).count()
    document_count = df.select("document_id").distinct().count()

    print("=" * 60)
    print(name)
    print(f"Chunks:        {total:,}")
    print(f"Documents:     {document_count}")
    print(f"Null IDs:      {null_ids}")
    print(f"Empty text:    {empty_text}")
    print(f"Duplicate IDs: {duplicate_ids}")

    if null_ids != 0:
        raise RuntimeError(f"{name}: null chunk IDs detected.")
    if empty_text != 0:
        raise RuntimeError(f"{name}: empty chunks detected.")
    if duplicate_ids != 0:
        raise RuntimeError(f"{name}: duplicate IDs detected.")
    if document_count != 5:
        raise RuntimeError(f"{name}: expected 5 documents, found {document_count}.")
    print("Validation passed.")

validate_chunk_dataframe(fixed_df, "V1 Fixed Size")
validate_chunk_dataframe(recursive_df, "V2 Recursive + Overlap")
validate_chunk_dataframe(structure_df, "V3 Structure-Aware Parent-Child")

# COMMAND ----------

# Structural metrics. These describe chunk behavior, NOT retrieval quality.
def calculate_chunk_metrics(df):
    stats = df.agg(
        F.count("*").alias("chunk_count"),
        F.avg("character_count").alias("avg_chars"),
        F.expr("percentile_approx(character_count, 0.5)").alias("median_chars"),
        F.expr("percentile_approx(character_count, 0.9)").alias("p90_chars"),
        F.expr("percentile_approx(character_count, 0.95)").alias("p95_chars"),
        F.min("character_count").alias("min_chars"),
        F.max("character_count").alias("max_chars"),
        F.avg("approx_token_count").alias("avg_approx_tokens"),
        F.sum(F.when(F.col("character_count") < MIN_QUALITY_CHARS, 1).otherwise(0)).alias("tiny_chunks"),
        F.sum(F.when(F.col("character_count") > MAX_CHARS, 1).otherwise(0)).alias("oversized_chunks"),
    ).first()

    chunk_count = int(stats["chunk_count"])
    tiny_chunks = int(stats["tiny_chunks"])
    return {
        "chunk_count": chunk_count,
        "avg_chars": float(stats["avg_chars"]),
        "median_chars": float(stats["median_chars"]),
        "p90_chars": float(stats["p90_chars"]),
        "p95_chars": float(stats["p95_chars"]),
        "min_chars": float(stats["min_chars"]),
        "max_chars": float(stats["max_chars"]),
        "avg_approx_tokens": float(stats["avg_approx_tokens"]),
        "tiny_chunks": tiny_chunks,
        "tiny_chunk_rate": tiny_chunks / chunk_count,
        "oversized_chunks": int(stats["oversized_chunks"]),
    }

fixed_metrics = calculate_chunk_metrics(fixed_df)
recursive_metrics = calculate_chunk_metrics(recursive_df)
structure_metrics = calculate_chunk_metrics(structure_df)

print("V1 FIXED\n", json.dumps(fixed_metrics, indent=2))
print("\nV2 RECURSIVE\n", json.dumps(recursive_metrics, indent=2))
print("\nV3 STRUCTURE\n", json.dumps(structure_metrics, indent=2))

# COMMAND ----------

# Comparison table
comparison_rows = [
    ("V1", "fixed_size", fixed_metrics["chunk_count"], fixed_metrics["avg_chars"],
     fixed_metrics["median_chars"], fixed_metrics["p90_chars"], fixed_metrics["tiny_chunks"],
     fixed_metrics["tiny_chunk_rate"], fixed_metrics["oversized_chunks"]),
    ("V2", "recursive_overlap", recursive_metrics["chunk_count"], recursive_metrics["avg_chars"],
     recursive_metrics["median_chars"], recursive_metrics["p90_chars"], recursive_metrics["tiny_chunks"],
     recursive_metrics["tiny_chunk_rate"], recursive_metrics["oversized_chunks"]),
    ("V3", "structure_aware_parent_child", structure_metrics["chunk_count"], structure_metrics["avg_chars"],
     structure_metrics["median_chars"], structure_metrics["p90_chars"], structure_metrics["tiny_chunks"],
     structure_metrics["tiny_chunk_rate"], structure_metrics["oversized_chunks"]),
]

comparison_schema = """
strategy STRING,
chunking_method STRING,
chunk_count LONG,
avg_chars DOUBLE,
median_chars DOUBLE,
p90_chars DOUBLE,
tiny_chunks LONG,
tiny_chunk_rate DOUBLE,
oversized_chunks LONG
"""

comparison_df = spark.createDataFrame(comparison_rows, schema=comparison_schema)
display(comparison_df)

# COMMAND ----------

# Distribution by report year
def report_distribution(df, strategy_name):
    return (
        df.groupBy("report_year")
        .agg(
            F.count("*").alias("chunks"),
            F.avg("character_count").alias("avg_chars"),
            F.expr("percentile_approx(character_count, 0.5)").alias("median_chars"),
        )
        .withColumn("strategy", F.lit(strategy_name))
    )

report_comparison_df = (
    report_distribution(fixed_df, "V1_fixed")
    .unionByName(report_distribution(recursive_df, "V2_recursive"))
    .unionByName(report_distribution(structure_df, "V3_structure"))
    .orderBy("report_year", "strategy")
)
display(report_comparison_df)

# COMMAND ----------

# Inspect the smallest chunks from each strategy.
print("V1 tiny chunks")
display(
    fixed_df.filter(F.col("character_count") < MIN_QUALITY_CHARS)
    .select("report_year", "page_start", "character_count", "chunk_text")
    .orderBy("character_count").limit(20)
)

print("V2 tiny chunks")
display(
    recursive_df.filter(F.col("character_count") < MIN_QUALITY_CHARS)
    .select("report_year", "page_start", "character_count", "chunk_text")
    .orderBy("character_count").limit(20)
)

print("V3 tiny chunks")
display(
    structure_df.filter(F.col("character_count") < MIN_QUALITY_CHARS)
    .select("report_year", "page_start", "character_count", "chunk_text")
    .orderBy("character_count").limit(20)
)

# COMMAND ----------

# Persist experimental candidates
for df, table_name in [
    (fixed_df, FIXED_TABLE),
    (recursive_df, RECURSIVE_TABLE),
    (structure_df, STRUCTURE_TABLE),
]:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    print(f"Saved: {table_name}")

# COMMAND ----------

# MLflow logging helper
def log_chunking_run(run_name, method, version, metrics, params):
    """Log one chunking strategy as one MLflow run."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "project": "global-economic-prospects-agent",
            "experiment_stage": "chunking",
            "corpus": "world-bank-gep-2022-2026",
            "strategy": method,
        })
        mlflow.log_param("chunking_method", method)
        mlflow.log_param("chunking_version", version)
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        print(f"Logged run: {run_name}")
        print(f"Run ID: {run.info.run_id}")
        return run.info.run_id

# COMMAND ----------

# Log V1, V2, V3
fixed_run_id = log_chunking_run(
    run_name="V1_fixed_size",
    method="fixed_size",
    version="v1",
    metrics=fixed_metrics,
    params={
        "target_chars": TARGET_CHARS,
        "overlap_chars": 0,
        "max_chars": TARGET_CHARS,
        "structure_aware": False,
        "parent_child": False,
    },
)

recursive_run_id = log_chunking_run(
    run_name="V2_recursive_overlap",
    method="recursive_overlap",
    version="v1",
    metrics=recursive_metrics,
    params={
        "target_chars": TARGET_CHARS,
        "max_chars": MAX_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "structure_aware": False,
        "parent_child": False,
    },
)

# These are the actual parameters used by the existing V3 baseline.
structure_run_id = log_chunking_run(
    run_name="V3_structure_parent_child",
    method="structure_aware_parent_child",
    version="baseline_v1",
    metrics=structure_metrics,
    params={
        "target_chars": 2500,
        "max_chars": 3500,
        "overlap_chars": 300,
        "structure_aware": True,
        "parent_child": True,
    },
)

# COMMAND ----------

# Persist experiment summary for SQL/dashboarding
summary_df = (
    comparison_df
    .withColumn("experiment_name", F.lit(EXPERIMENT_NAME))
    .withColumn("experiment_timestamp", F.current_timestamp())
)

(
    summary_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SUMMARY_TABLE)
)
print(f"Saved experiment summary: {SUMMARY_TABLE}")

# COMMAND ----------

# Read-back validation
saved_fixed_df = spark.table(FIXED_TABLE)
saved_recursive_df = spark.table(RECURSIVE_TABLE)
saved_structure_df = spark.table(STRUCTURE_TABLE)

saved_fixed_count = saved_fixed_df.count()
saved_recursive_count = saved_recursive_df.count()
saved_structure_count = saved_structure_df.count()

assert saved_fixed_count == fixed_metrics["chunk_count"]
assert saved_recursive_count == recursive_metrics["chunk_count"]
assert saved_structure_count == structure_metrics["chunk_count"]

print("Persisted-table validation passed.")
print(f"V1 fixed:      {saved_fixed_count:,}")
print(f"V2 recursive:  {saved_recursive_count:,}")
print(f"V3 structure:  {saved_structure_count:,}")

# COMMAND ----------

# Final comparison
display(spark.table(SUMMARY_TABLE).orderBy("strategy"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretation
# MAGIC Structural metrics such as chunk count, size distribution, and tiny-chunk rate are diagnostics only. Do not choose the production strategy from these metrics alone. The next stage should create a retrieval evaluation set and measure evidence retrieval quality (for example Recall@K and MRR) before selecting a winner.

# COMMAND ----------

print("=" * 72)
print("01_chunking_experiments COMPLETED")
print("=" * 72)
print(f"V1 Fixed Size:             {saved_fixed_count:,}")
print(f"V2 Recursive + Overlap:    {saved_recursive_count:,}")
print(f"V3 Structure Parent-Child: {saved_structure_count:,}")
print()
print("MLflow runs:")
print(f"V1: {fixed_run_id}")
print(f"V2: {recursive_run_id}")
print(f"V3: {structure_run_id}")
print()
print("No winning chunking strategy has been selected yet.")
print("Next: build the retrieval evaluation dataset, then evaluate Recall@K / MRR.")
print("=" * 72)