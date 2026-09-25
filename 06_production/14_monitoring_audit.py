# Databricks notebook source
# MAGIC %md
# MAGIC # 14 - Production monitoring and audit queries
# MAGIC Use this after the endpoint and Databricks App are receiving traffic.

# COMMAND ----------

from pyspark.sql import functions as F

APP_LOGS = "worldbank_ai.monitoring.application_logs"
FEEDBACK = "worldbank_ai.monitoring.user_feedback"

# COMMAND ----------

# Traffic, errors, and latency by route.
display(
    spark.table(APP_LOGS)
    .groupBy("route")
    .agg(
        F.count("*").alias("request_count"),
        F.avg("latency_ms").alias("avg_latency_ms"),
        F.expr("percentile_approx(latency_ms, 0.95)").alias("p95_latency_ms"),
        F.sum(F.when(F.col("status") != "success", 1).otherwise(0)).alias("error_count"),
    )
    .orderBy(F.desc("request_count"))
)

# COMMAND ----------

# Daily production health.
display(
    spark.table(APP_LOGS)
    .withColumn("day", F.to_date("created_at"))
    .groupBy("day")
    .agg(
        F.count("*").alias("requests"),
        F.avg("latency_ms").alias("avg_latency_ms"),
        F.sum(F.when(F.col("status") != "success", 1).otherwise(0)).alias("errors"),
    )
    .orderBy(F.desc("day"))
)

# COMMAND ----------

# User feedback distribution.
display(
    spark.table(FEEDBACK)
    .groupBy("rating")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# Recent failures for investigation.
display(
    spark.table(APP_LOGS)
    .filter(F.col("status") != "success")
    .orderBy(F.desc("created_at"))
    .limit(100)
)
