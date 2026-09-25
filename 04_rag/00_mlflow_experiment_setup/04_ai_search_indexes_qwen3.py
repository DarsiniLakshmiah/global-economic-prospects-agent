# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04_ai_search_indexes_qwen3
# MAGIC
# MAGIC Purpose:
# MAGIC - Create/reuse one STANDARD Databricks AI Search endpoint
# MAGIC - Create three Delta Sync indexes for V1/V2/V3
# MAGIC - Use the same embedding model for all three chunking strategies
# MAGIC - Trigger sync and wait for indexes to become ONLINE
# MAGIC - Run smoke-test searches
# MAGIC
# MAGIC **You can use Run All.**
# MAGIC
# MAGIC This notebook is idempotent: existing endpoint/index resources are reused.
# MAGIC
# MAGIC Embedding model: `databricks-qwen3-embedding-0-6b`
# MAGIC
# MAGIC We use one embedding model here so the first experiment isolates the
# MAGIC effect of chunking strategy. We compare embedding models only after
# MAGIC choosing the strongest chunking/retrieval configuration.

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-ai-search

# COMMAND ----------

# ============================================================
# 01. Imports and configuration
# ============================================================

import time
from pyspark.sql import functions as F
from databricks.ai_search.client import AISearchClient

CATALOG = "worldbank_ai"
RAG_SCHEMA = "rag"

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
ENDPOINT_TYPE = "STANDARD"

# Current Databricks-recommended hosted embedding model for
# production AI Search on standard endpoints.
EMBEDDING_MODEL = "databricks-qwen3-embedding-0-6b"

SOURCE_TABLES = {
    "fixed_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_retrieval_fixed_v1",
    "recursive_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_retrieval_recursive_v1",
    "structure_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_retrieval_structure_v1",
}

INDEX_NAMES = {
    "fixed_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_fixed_v1_qwen3_index",
    "recursive_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_recursive_v1_qwen3_index",
    "structure_v1": f"{CATALOG}.{RAG_SCHEMA}.gep_structure_v1_qwen3_index",
}

COLUMNS_TO_SYNC = [
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
    "chunking_strategy",
]

client = AISearchClient()

print("Configuration loaded.")
print(f"AI Search endpoint: {AI_SEARCH_ENDPOINT}")
print(f"Embedding model:    {EMBEDDING_MODEL}")

# COMMAND ----------

# ============================================================
# 02. Validate canonical source tables
# ============================================================

for strategy, table_name in SOURCE_TABLES.items():
    assert spark.catalog.tableExists(table_name), (
        f"Missing canonical source table: {table_name}. "
        "Run 03_prepare_retrieval_corpora first."
    )

    df = spark.table(table_name)

    assert df.count() > 0
    assert df.filter(F.col("chunk_id").isNull()).count() == 0
    assert df.groupBy("chunk_id").count().filter("count > 1").count() == 0
    assert df.filter(
        F.col("retrieval_text").isNull()
        | (F.length(F.trim("retrieval_text")) == 0)
    ).count() == 0

    print(f"{strategy}: source validation passed ({df.count()} rows).")

# COMMAND ----------

# ============================================================
# 03. Create or reuse AI Search endpoint
# ============================================================

def get_endpoint_if_exists(name):
    try:
        return client.get_endpoint(name=name)
    except Exception:
        return None

endpoint = get_endpoint_if_exists(AI_SEARCH_ENDPOINT)

if endpoint is None:
    print(f"Creating AI Search endpoint: {AI_SEARCH_ENDPOINT}")
    client.create_endpoint(
        name=AI_SEARCH_ENDPOINT,
        endpoint_type=ENDPOINT_TYPE,
    )
else:
    print(f"Reusing existing AI Search endpoint: {AI_SEARCH_ENDPOINT}")

# COMMAND ----------

# ============================================================
# Diagnose AI Search endpoint + index states
# ============================================================
# DO NOT delete or recreate anything yet.

import json

print("=" * 70)
print("AI SEARCH ENDPOINT")
print("=" * 70)

endpoint_info = client.get_endpoint(
    name=AI_SEARCH_ENDPOINT
)

print(
    json.dumps(
        endpoint_info,
        indent=2,
        default=str
    )
)


print("\n" + "=" * 70)
print("INDEX STATES")
print("=" * 70)

for strategy, index_name in INDEX_NAMES.items():

    print(f"\n--- {strategy} ---")

    try:
        index = client.get_index(
            endpoint_name=AI_SEARCH_ENDPOINT,
            index_name=index_name,
        )

        description = index.describe()

        # Only print the parts we need for diagnosis.
        print(
            json.dumps(
                {
                    "name": index_name,
                    "status": description.get("status"),
                    "delta_sync_index_spec":
                        description.get("delta_sync_index_spec"),
                },
                indent=2,
                default=str
            )
        )

    except Exception as e:

        print(
            f"{type(e).__name__}: {e}"
        )

# COMMAND ----------

# ============================================================
# 07. Load index handles + verify final index readiness
# ============================================================
# This cell is self-contained.
#
# It recreates the Python `indexes` dictionary from the
# already-existing Databricks AI Search indexes.
#
# It does NOT create, delete, or sync any indexes.
# ============================================================


# ------------------------------------------------------------
# Create the local dictionary
# ------------------------------------------------------------

indexes = {}


# ------------------------------------------------------------
# Load each existing index and verify readiness
# ------------------------------------------------------------

for strategy, index_name in INDEX_NAMES.items():

    print(f"\nChecking index: {strategy}")

    # Get the existing Databricks AI Search index
    index = client.get_index(
        endpoint_name=AI_SEARCH_ENDPOINT,
        index_name=index_name,
    )

    # Save the handle for later cells
    indexes[strategy] = index

    # Get current server-side status
    description = index.describe()

    status = description.get(
        "status",
        {}
    )

    ready = status.get("ready")
    indexed_rows = status.get("indexed_row_count")
    detailed_state = status.get("detailed_state")
    message = status.get("message")

    print(f"Ready:          {ready}")
    print(f"Indexed rows:   {indexed_rows}")
    print(f"Detailed state: {detailed_state}")
    print(f"Message:        {message}")

    # Hard validation
    assert ready is True, (
        f"{strategy} index is not ready. "
        f"Current state: {detailed_state}"
    )


print("\nAll three AI Search indexes are READY.")
print("Index handles loaded into `indexes`.")

# COMMAND ----------

# ============================================================
# 06. Trigger sync for all indexes
# ============================================================
# Triggered sync is appropriate for this controlled experiment.
# It avoids paying for a continuously running sync pipeline.

for strategy, index in indexes.items():
    print(f"Triggering sync: {strategy}")
    try:
        index.sync()
    except Exception as e:
        # A newly-created index can already be syncing.
        # We do not hide the message; we continue to readiness polling.
        print(f"Sync call returned: {type(e).__name__}: {e}")

# COMMAND ----------

# ============================================================
# 07. Wait until all indexes are ONLINE
# ============================================================

INDEX_TIMEOUT_SECONDS = 45 * 60

def detailed_state(index):
    desc = index.describe()
    return str(
        desc.get("status", {}).get("detailed_state")
        or desc.get("status", {}).get("state")
        or ""
    ).upper(), desc

for strategy, index in indexes.items():
    print(f"\nWaiting for {strategy} index...")
    start = time.time()

    while True:
        state, desc = detailed_state(index)
        print(f"{strategy}: {state or 'UNKNOWN'}")

        if state.startswith("ONLINE"):
            break

        if "FAIL" in state or "ERROR" in state:
            raise RuntimeError(
                f"{strategy} index entered failure state: {desc}"
            )

        if time.time() - start > INDEX_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"{strategy} index did not become ONLINE within "
                f"{INDEX_TIMEOUT_SECONDS // 60} minutes."
            )

        time.sleep(15)

print("\nAll three indexes are ONLINE.")

# COMMAND ----------

# ============================================================
# 08. Refresh index handles and inspect configurations
# ============================================================

for strategy, index_name in INDEX_NAMES.items():
    indexes[strategy] = client.get_index(
        endpoint_name=AI_SEARCH_ENDPOINT,
        index_name=index_name,
    )

    print(f"\n{strategy}")
    print(indexes[strategy].describe())

# COMMAND ----------

# ============================================================
# 09. ANN smoke test
# ============================================================

TEST_QUERY = (
    "What risks did the World Bank identify for the global economy?"
)

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

for strategy, index in indexes.items():
    print(f"\nANN smoke test: {strategy}")

    result = index.similarity_search(
        query_text=TEST_QUERY,
        columns=RETURN_COLUMNS,
        num_results=3,
        query_type="ANN",
    )

    rows = result.get("result", {}).get("data_array", [])
    assert len(rows) > 0, f"{strategy}: ANN search returned no results"

    print(f"Returned {len(rows)} rows.")

# COMMAND ----------

# ============================================================
# 10. HYBRID + metadata-filter smoke test
# ============================================================

for strategy, index in indexes.items():
    print(f"\nHYBRID filtered smoke test: {strategy}")

    result = index.similarity_search(
        query_text="What risks did the World Bank identify in 2025?",
        columns=RETURN_COLUMNS,
        num_results=5,
        query_type="HYBRID",
        filters={"report_year": 2025},
    )

    rows = result.get("result", {}).get("data_array", [])
    assert len(rows) > 0, f"{strategy}: filtered hybrid search returned no results"

    print(f"Returned {len(rows)} rows.")

print("\n04_ai_search_indexes_qwen3 COMPLETE.")
print("Next: 05_retrieval_evaluation")