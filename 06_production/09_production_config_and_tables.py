# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 09 - Production configuration and monitoring tables
# MAGIC Run this once before model packaging.

# COMMAND ----------

CATALOG = "worldbank_ai"
AI_SCHEMA = "ai"
MONITORING_SCHEMA = "monitoring"

MODEL_NAME = f"{CATALOG}.{AI_SCHEMA}.gep_intelligence_agent"
ENDPOINT_NAME = "worldbank-gep-intelligence-agent"

print("Model:", MODEL_NAME)
print("Endpoint:", ENDPOINT_NAME)

# COMMAND ----------

# SQL Warehouse used by the production Data Agent.
dbutils.widgets.text(
    "sql_warehouse_id",
    "3d11225dd32e8158"
)

SQL_WAREHOUSE_ID = dbutils.widgets.get(
    "sql_warehouse_id"
).strip()

assert SQL_WAREHOUSE_ID, (
    "Enter the SQL warehouse ID in the sql_warehouse_id widget."
)

print("SQL warehouse configured.")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}.application_logs (
  request_id STRING,
  session_id STRING,
  user_name STRING,
  question STRING,
  route STRING,
  status STRING,
  latency_ms DOUBLE,
  endpoint_name STRING,
  created_at TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}.user_feedback (
  request_id STRING,
  session_id STRING,
  rating STRING,
  feedback_text STRING,
  created_at TIMESTAMP
) USING DELTA
""")

print("Monitoring tables READY.")