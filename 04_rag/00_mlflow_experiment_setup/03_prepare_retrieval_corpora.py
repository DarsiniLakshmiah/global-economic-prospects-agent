# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03_prepare_retrieval_corpora
# MAGIC
# MAGIC Purpose:
# MAGIC - Normalize V1 fixed-size, V2 recursive, and V3 structure-aware chunk tables
# MAGIC - Preserve source provenance
# MAGIC - Create three canonical Delta source tables for AI Search
# MAGIC - Enable Change Data Feed (required for STANDARD Delta Sync indexes)
# MAGIC - Validate IDs, text, years, pages, and duplicates
# MAGIC
# MAGIC Safe to use **Run All**.
# MAGIC
# MAGIC This notebook does NOT call an LLM and does NOT create vector indexes.

# COMMAND ----------

# ============================================================
# 01. Configuration
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import types as T

CATALOG = "worldbank_ai"
RAG_SCHEMA = "rag"

SOURCE_TABLES = {
    "fixed_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_chunks_fixed_v1",
    "recursive_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_chunks_recursive_v1",
    "structure_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_chunks_structure_v1",
}

CANONICAL_TABLES = {
    "fixed_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_retrieval_fixed_v1",
    "recursive_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_retrieval_recursive_v1",
    "structure_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_retrieval_structure_v1",
}

EXPECTED_YEARS = [2022, 2023, 2024, 2025, 2026]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{RAG_SCHEMA}")

print("Configuration loaded.")
for k, v in SOURCE_TABLES.items():
    print(f"{k:15s} -> {v}")

# COMMAND ----------

# ============================================================
# 02. Inspect source schemas
# ============================================================

for strategy, table_name in SOURCE_TABLES.items():
    assert spark.catalog.tableExists(table_name), f"Missing source table: {table_name}"
    df = spark.table(table_name)
    print(f"\n{strategy}: {df.count()} rows")
    print(df.columns)

# COMMAND ----------

# ============================================================
# 03. Canonicalization helpers
# ============================================================
# Different chunking experiments may have slightly different
# schemas. We map them into one stable retrieval schema.
#
# We do NOT fabricate source provenance. If a source field does
# not exist, it remains NULL except edition_status, which is
# deterministic from the known GEP edition year.

def first_existing(df, candidates, dtype="string"):
    """Return the first existing source column, otherwise typed NULL."""
    for c in candidates:
        if c in df.columns:
            return F.col(c).cast(dtype)
    return F.lit(None).cast(dtype)

def canonicalize(df, strategy):
    # Find the text column used by this experimental corpus.
    text_expr = first_existing(
        df,
        ["chunk_text", "text", "content", "retrieval_text"],
        "string"
    )

    # Use existing retrieval_text only when present; otherwise
    # use the actual chunk text. We do not inject guessed metadata.
    retrieval_expr = (
        F.col("retrieval_text").cast("string")
        if "retrieval_text" in df.columns
        else text_expr
    )

    report_year = first_existing(df, ["report_year", "year"], "int")

    result = df.select(
        first_existing(df, ["chunk_id", "id"], "string").alias("chunk_id"),
        first_existing(df, ["parent_chunk_id", "parent_id"], "string").alias("parent_chunk_id"),
        first_existing(df, ["document_id", "doc_id"], "string").alias("document_id"),
        report_year.alias("report_year"),
        first_existing(df, ["edition_status"], "string").alias("_edition_status"),
        first_existing(df, ["chapter", "chapter_title"], "string").alias("chapter"),
        first_existing(df, ["region"], "string").alias("region"),
        first_existing(df, ["section"], "string").alias("section"),
        first_existing(df, ["subsection"], "string").alias("subsection"),
        first_existing(df, ["content_type"], "string").alias("content_type"),
        first_existing(df, ["page_start", "start_page", "page_number"], "int").alias("page_start"),
        first_existing(df, ["page_end", "end_page", "page_number"], "int").alias("page_end"),
        text_expr.alias("chunk_text"),
        retrieval_expr.alias("retrieval_text"),
    )

    # Known report-edition rule from this project:
    # Jan 2022-2025 = final; Jan 2026 = advance.
    result = (
        result
        .withColumn(
            "edition_status",
            F.coalesce(
                F.col("_edition_status"),
                F.when(F.col("report_year") == 2026, F.lit("advance"))
                 .when(F.col("report_year").isin(2022, 2023, 2024, 2025), F.lit("final"))
            )
        )
        .drop("_edition_status")
        .withColumn("chunking_strategy", F.lit(strategy))
        .withColumn("prepared_at", F.current_timestamp())
    )

    return result

# COMMAND ----------

# ============================================================
# 04. Build canonical retrieval tables
# ============================================================

prepared = {}

for strategy, source_table in SOURCE_TABLES.items():
    target_table = CANONICAL_TABLES[strategy]

    source_df = spark.table(source_table)
    canonical_df = canonicalize(source_df, strategy)

    (
        canonical_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )

    # STANDARD AI Search Delta Sync indexes require CDF.
    spark.sql(
        f"ALTER TABLE {target_table} "
        f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    prepared[strategy] = spark.table(target_table)

    print(
        f"{strategy}: wrote {prepared[strategy].count()} rows -> {target_table}"
    )

# COMMAND ----------

# ============================================================
# 05. Hard validation
# ============================================================

for strategy, df in prepared.items():
    count = df.count()
    assert count > 0, f"{strategy}: table is empty"

    # Primary key must exist and be unique.
    null_ids = df.filter(F.col("chunk_id").isNull()).count()
    dup_ids = (
        df.groupBy("chunk_id")
          .count()
          .filter(F.col("count") > 1)
          .count()
    )

    assert null_ids == 0, f"{strategy}: NULL chunk_id rows = {null_ids}"
    assert dup_ids == 0, f"{strategy}: duplicate chunk_id values = {dup_ids}"

    # Retrieval text must be usable.
    bad_text = df.filter(
        F.col("retrieval_text").isNull()
        | (F.length(F.trim(F.col("retrieval_text"))) == 0)
    ).count()
    assert bad_text == 0, f"{strategy}: empty retrieval_text rows = {bad_text}"

    # All five GEP editions must be represented.
    years = sorted(
        r["report_year"]
        for r in df.select("report_year").distinct().collect()
        if r["report_year"] is not None
    )
    assert years == EXPECTED_YEARS, f"{strategy}: unexpected years {years}"

    # Document ID is required for strategy-neutral evaluation.
    missing_doc = df.filter(F.col("document_id").isNull()).count()
    assert missing_doc == 0, f"{strategy}: missing document_id rows = {missing_doc}"

    print(f"{strategy}: hard validation passed ({count} rows).")

# COMMAND ----------

# ============================================================
# 06. Provenance diagnostics
# ============================================================
# Page provenance is important because the final benchmark uses
# document + page overlap instead of V3 chunk IDs when comparing
# V1/V2/V3.

for strategy, df in prepared.items():
    stats = df.agg(
        F.count("*").alias("rows"),
        F.sum(F.col("page_start").isNull().cast("int")).alias("missing_page_start"),
        F.sum(F.col("page_end").isNull().cast("int")).alias("missing_page_end"),
        F.avg(F.length("retrieval_text")).alias("avg_chars"),
        F.expr("percentile_approx(length(retrieval_text), 0.5)").alias("median_chars"),
        F.expr("percentile_approx(length(retrieval_text), 0.9)").alias("p90_chars"),
    )
    print(f"\n{strategy}")
    display(stats)

# COMMAND ----------

# ============================================================
# 07. Distribution checks
# ============================================================

for strategy, df in prepared.items():
    print(f"\n{strategy} by report year")
    display(
        df.groupBy("report_year")
          .count()
          .orderBy("report_year")
    )

# COMMAND ----------

# ============================================================
# 08. Confirm Change Data Feed
# ============================================================

for strategy, table_name in CANONICAL_TABLES.items():
    props = spark.sql(f"SHOW TBLPROPERTIES {table_name}")
    cdf = (
        props.filter(F.col("key") == "delta.enableChangeDataFeed")
             .select("value")
             .collect()
    )
    assert cdf and cdf[0]["value"].lower() == "true", (
        f"{strategy}: Change Data Feed is not enabled"
    )
    print(f"{strategy}: Change Data Feed enabled.")

# COMMAND ----------

# ============================================================
# 09. Final summary
# ============================================================

summary_rows = []

for strategy, table_name in CANONICAL_TABLES.items():
    df = spark.table(table_name)
    summary_rows.append(
        (
            strategy,
            table_name,
            df.count(),
            df.filter(F.col("page_start").isNull()).count(),
            df.filter(F.col("page_end").isNull()).count(),
        )
    )

summary_df = spark.createDataFrame(
    summary_rows,
    [
        "strategy",
        "canonical_table",
        "row_count",
        "missing_page_start",
        "missing_page_end",
    ]
)

display(summary_df)

print("\n03_prepare_retrieval_corpora COMPLETE.")
print("Next: 04_ai_search_indexes_qwen3")