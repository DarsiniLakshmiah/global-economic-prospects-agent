# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 08_agent_evaluation_v3
# MAGIC
# MAGIC End-to-end evaluation for the World Bank intelligence agent.
# MAGIC
# MAGIC Order is intentional:
# MAGIC 1. install shared runtime packages once
# MAGIC 2. restart Python once
# MAGIC 3. import runtime notebooks
# MAGIC 4. load/seed the governed benchmark
# MAGIC 5. execute and persist evaluation results

# COMMAND ----------

# Install dependencies once. Runtime notebooks themselves never install packages.
%pip install -q databricks-openai databricks-ai-search

# COMMAND ----------

# Restart once after package installation.
# Everything required later is defined AFTER this restart.
dbutils.library.restartPython()

# COMMAND ----------

# Imports MUST be after restartPython().
from pyspark.sql import functions as F, types as T
from datetime import datetime, timezone
from databricks_openai import DatabricksOpenAI
from databricks.ai_search.client import AISearchClient
import json
import time
import uuid

print("Runtime dependencies READY.")
print(" - databricks-openai: READY")
print(" - databricks-ai-search: READY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load definition-only runtime notebooks

# COMMAND ----------

# MAGIC %run ./runtime/supervisor_runtime

# COMMAND ----------

# MAGIC %run ./runtime/data_agent_runtime

# COMMAND ----------

# MAGIC %run ./runtime/research_agent_runtime

# COMMAND ----------

# MAGIC %run ./runtime/synthesis_agent_runtime

# COMMAND ----------

# MAGIC %run ./runtime/guardrails_runtime

# COMMAND ----------

# Verify all runtime entry points AFTER all %run calls.
required_runtime_functions = [
    "run_supervisor_agent",
    "run_data_agent",
    "run_research_agent",
    "run_synthesis_agent",
    "apply_pre_execution_guardrails",
    "apply_post_execution_guardrails",
]

missing_runtime_functions = [
    name
    for name in required_runtime_functions
    if name not in globals() or not callable(globals()[name])
]

if missing_runtime_functions:
    raise RuntimeError(
        "Runtime layer is incomplete. Missing functions: "
        f"{missing_runtime_functions}"
    )

print("Runtime layer READY.")
for name in required_runtime_functions:
    print(f" - {name}: READY")


# Verify that the shared Databricks %run namespace is safe.
# The Supervisor's typed validator and the Guardrails dict validator must
# have different names.
assert "validate_supervisor_output" in globals()
assert callable(validate_supervisor_output)

assert "validate_execution_plan" in globals()
assert callable(validate_execution_plan)

print("Runtime namespace collision checks PASSED.")

# COMMAND ----------

# Governed evaluation tables.
CATALOG = "worldbank_ai"
AI_SCHEMA = "ai"

EVAL_TABLE = f"{CATALOG}.{AI_SCHEMA}.agent_evaluation_dataset"
RESULTS_TABLE = f"{CATALOG}.{AI_SCHEMA}.agent_evaluation_results"
SUMMARY_TABLE = f"{CATALOG}.{AI_SCHEMA}.agent_evaluation_summary"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{AI_SCHEMA}")

eval_schema = T.StructType([
    T.StructField("eval_id", T.StringType(), False),
    T.StructField("question", T.StringType(), False),
    T.StructField("expected_route", T.StringType(), False),
    T.StructField("expected_report_years", T.ArrayType(T.IntegerType()), True),
    T.StructField("expected_entities", T.ArrayType(T.StringType()), True),
    T.StructField("expected_indicator_codes", T.ArrayType(T.StringType()), True),
    T.StructField("reference_notes", T.StringType(), True),
    T.StructField("is_approved", T.BooleanType(), False),
])

# COMMAND ----------

# Explicit benchmark definitions.
# These rows contain questions and expected routing metadata only.
# They do NOT contain fabricated model answers or fabricated evaluation results.

evaluation_cases = [
    # Structured
    ("agent_eval_001", "Show India's GDP growth from 2015 to 2025.", "structured", [], ["India"], ["NY.GDP.MKTP.KD.ZG"], "Historical GDP growth series only.", True),
    ("agent_eval_002", "Show GDP per capita growth in India from 2018 to 2025.", "structured", [], ["India"], ["NY.GDP.PCAP.KD.ZG"], "Historical GDP per capita growth.", True),
    ("agent_eval_003", "Show India's inflation rate from 2015 to 2025.", "structured", [], ["India"], ["FP.CPI.TOTL.ZG"], "Historical consumer price inflation.", True),
    ("agent_eval_004", "Compare GDP growth in India and China from 2020 to 2025.", "structured", [], ["India", "China"], ["NY.GDP.MKTP.KD.ZG"], "Two-country historical comparison.", True),
    ("agent_eval_005", "Compare GDP growth in South Asia and Sub-Saharan Africa from 2022 to 2025.", "structured", [], ["South Asia", "Sub-Saharan Africa"], ["NY.GDP.MKTP.KD.ZG"], "Historical regional comparison.", True),
    ("agent_eval_006", "Show trade as a percentage of GDP for India from 2015 to 2025.", "structured", [], ["India"], ["NE.TRD.GNFS.ZS"], "Historical trade indicator.", True),
    ("agent_eval_007", "Show foreign direct investment net inflows as a percentage of GDP for India from 2015 to 2025.", "structured", [], ["India"], ["BX.KLT.DINV.WD.GD.ZS"], "Historical FDI indicator.", True),
    ("agent_eval_008", "Show India's exports as a percentage of GDP from 2018 to 2025.", "structured", [], ["India"], ["NE.EXP.GNFS.ZS"], "Historical export indicator.", True),
    ("agent_eval_009", "Show India's imports as a percentage of GDP from 2018 to 2025.", "structured", [], ["India"], ["NE.IMP.GNFS.ZS"], "Historical import indicator.", True),
    ("agent_eval_010", "Show India's current account balance as a percentage of GDP from 2015 to 2025.", "structured", [], ["India"], ["BN.CAB.XOKA.GD.ZS"], "Historical current-account series.", True),

    # RAG
    ("agent_eval_011", "What risks did the World Bank identify for the global economy in the January 2025 Global Economic Prospects report?", "rag", [2025], [], [], "Single-edition qualitative GEP retrieval.", True),
    ("agent_eval_012", "What did the January 2025 Global Economic Prospects report say about the outlook for South Asia?", "rag", [2025], ["South Asia"], [], "Regional qualitative outlook.", True),
    ("agent_eval_013", "What did the January 2025 GEP say about the outlook for Sub-Saharan Africa?", "rag", [2025], ["Sub-Saharan Africa"], [], "Regional qualitative outlook.", True),
    ("agent_eval_014", "What were the main downside risks discussed in the January 2024 Global Economic Prospects report?", "rag", [2024], [], [], "Risk retrieval from 2024 edition.", True),
    ("agent_eval_015", "How did the January 2023 Global Economic Prospects report describe global growth prospects?", "rag", [2023], [], [], "Global qualitative outlook.", True),
    ("agent_eval_016", "What economic challenges were highlighted in the January 2022 Global Economic Prospects report?", "rag", [2022], [], [], "2022 report evidence.", True),
    ("agent_eval_017", "What does the January 2026 Global Economic Prospects report say about global economic risks?", "rag", [2026], [], [], "2026 advance-edition evidence.", True),
    ("agent_eval_018", "According to the January 2025 GEP, what factors were affecting growth in developing economies?", "rag", [2025], [], [], "Qualitative retrieval.", True),
    ("agent_eval_019", "What did the January 2024 GEP say about investment and growth prospects?", "rag", [2024], [], [], "Qualitative report retrieval.", True),
    ("agent_eval_020", "What policy challenges were discussed in the January 2025 Global Economic Prospects report?", "rag", [2025], [], [], "Policy-oriented qualitative retrieval.", True),

    # Temporal RAG
    ("agent_eval_021", "How did the World Bank's assessment of global economic risks change between the January 2022 and January 2026 GEP reports?", "temporal_rag", [2022, 2026], [], [], "Cross-edition risk comparison.", True),
    ("agent_eval_022", "Compare the global growth outlook in the January 2023 and January 2025 GEP reports.", "temporal_rag", [2023, 2025], [], [], "Cross-edition outlook comparison.", True),
    ("agent_eval_023", "How did the World Bank's outlook for South Asia change between the January 2024 and January 2025 GEP reports?", "temporal_rag", [2024, 2025], ["South Asia"], [], "Regional temporal comparison.", True),
    ("agent_eval_024", "How did the outlook for Sub-Saharan Africa change between the January 2024 and January 2026 GEP reports?", "temporal_rag", [2024, 2026], ["Sub-Saharan Africa"], [], "Regional temporal comparison.", True),
    ("agent_eval_025", "Compare the downside risks discussed in the January 2022 and January 2025 GEP reports.", "temporal_rag", [2022, 2025], [], [], "Temporal risk comparison.", True),
    ("agent_eval_026", "How did global growth concerns evolve from the January 2023 GEP to the January 2026 GEP?", "temporal_rag", [2023, 2026], [], [], "Temporal global outlook.", True),
    ("agent_eval_027", "Compare the World Bank's discussion of developing economy growth in the January 2024 and January 2025 GEP reports.", "temporal_rag", [2024, 2025], [], [], "Cross-edition developing-economy comparison.", True),
    ("agent_eval_028", "How did the policy challenges described by the World Bank change between the January 2022 and January 2025 GEP reports?", "temporal_rag", [2022, 2025], [], [], "Cross-edition policy comparison.", True),
    ("agent_eval_029", "Compare the World Bank's global economic outlook in January 2024 with January 2026.", "temporal_rag", [2024, 2026], [], [], "Temporal global outlook.", True),
    ("agent_eval_030", "How did the discussion of risks to emerging and developing economies change between the January 2023 and January 2025 GEP reports?", "temporal_rag", [2023, 2025], [], [], "Temporal EMDE comparison.", True),

    # Hybrid
    ("agent_eval_031", "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 and explain the World Bank's January 2025 outlook for the two regions.", "hybrid", [2025], ["South Asia", "Sub-Saharan Africa"], ["NY.GDP.MKTP.KD.ZG"], "Core hybrid use case.", True),
    ("agent_eval_032", "Show India's GDP growth from 2020 to 2025 and explain the country's outlook in the January 2025 GEP.", "hybrid", [2025], ["India"], ["NY.GDP.MKTP.KD.ZG"], "Historical series plus qualitative outlook.", True),
    ("agent_eval_033", "Compare India and China GDP growth from 2020 to 2025 and explain the relevant outlook discussed in the January 2025 GEP.", "hybrid", [2025], ["India", "China"], ["NY.GDP.MKTP.KD.ZG"], "Country comparison plus GEP context.", True),
    ("agent_eval_034", "Show South Asia's GDP growth from 2020 to 2025 and summarize the risks to its outlook in the January 2025 GEP.", "hybrid", [2025], ["South Asia"], ["NY.GDP.MKTP.KD.ZG"], "Regional historical data plus risks.", True),
    ("agent_eval_035", "Show Sub-Saharan Africa's GDP growth from 2020 to 2025 and explain its outlook in the January 2025 GEP.", "hybrid", [2025], ["Sub-Saharan Africa"], ["NY.GDP.MKTP.KD.ZG"], "Regional historical data plus outlook.", True),
    ("agent_eval_036", "Show India's inflation from 2020 to 2025 and explain the inflation-related economic risks discussed in the January 2025 GEP.", "hybrid", [2025], ["India"], ["FP.CPI.TOTL.ZG"], "Inflation data plus GEP evidence.", True),
    ("agent_eval_037", "Show India's trade as a percentage of GDP from 2020 to 2025 and explain the global trade outlook in the January 2025 GEP.", "hybrid", [2025], ["India"], ["NE.TRD.GNFS.ZS"], "Trade data plus GEP outlook.", True),
    ("agent_eval_038", "Compare GDP growth in South Asia and Sub-Saharan Africa from 2022 to 2025 and explain how their outlook differs in the January 2026 GEP.", "hybrid", [2026], ["South Asia", "Sub-Saharan Africa"], ["NY.GDP.MKTP.KD.ZG"], "Historical data plus 2026 advance-edition outlook.", True),
    ("agent_eval_039", "Show India's GDP per capita growth from 2020 to 2025 and explain the development outlook discussed in the January 2025 GEP.", "hybrid", [2025], ["India"], ["NY.GDP.PCAP.KD.ZG"], "GDP per capita plus GEP context.", True),
    ("agent_eval_040", "Show India's foreign direct investment as a percentage of GDP from 2020 to 2025 and explain the investment outlook discussed in the January 2025 GEP.", "hybrid", [2025], ["India"], ["BX.KLT.DINV.WD.GD.ZS"], "FDI data plus investment outlook.", True),
]

seed_df = spark.createDataFrame(evaluation_cases, schema=eval_schema)

assert seed_df.count() == 40

route_counts = {
    row["expected_route"]: row["count"]
    for row in seed_df.groupBy("expected_route").count().collect()
}

assert route_counts == {
    "structured": 10,
    "rag": 10,
    "temporal_rag": 10,
    "hybrid": 10,
}

# Only initialize/repair the benchmark when the table is absent or has no approved rows.
# Existing approved benchmark rows are preserved.
if not spark.catalog.tableExists(EVAL_TABLE):
    seed_df.write.format("delta").mode("overwrite").saveAsTable(EVAL_TABLE)
    print(f"Created governed benchmark: {EVAL_TABLE}")
else:
    existing_approved_count = (
        spark.table(EVAL_TABLE)
        .filter(F.col("is_approved") == True)
        .count()
    )

    if existing_approved_count == 0:
        seed_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(EVAL_TABLE)
        print(f"Initialized empty benchmark with 40 approved cases: {EVAL_TABLE}")
    else:
        print(
            "Existing approved benchmark preserved:",
            existing_approved_count,
            "cases",
        )

# COMMAND ----------

# IMPORTANT: load the DataFrame AFTER restartPython and AFTER table initialization.
approved_eval_df = (
    spark.table(EVAL_TABLE)
    .filter(F.col("is_approved") == True)
)

approved_count = approved_eval_df.count()

print("Approved end-to-end evaluation cases:", approved_count)

if approved_count == 0:
    raise RuntimeError(
        f"{EVAL_TABLE} contains no approved evaluation cases."
    )

eval_df = approved_eval_df

print("Evaluation dataset READY.")
print("Evaluation cases:", eval_df.count())

display(
    eval_df
    .groupBy("expected_route")
    .count()
    .orderBy("expected_route")
)

# COMMAND ----------

def _extract_allowed_evidence_ids(research_result):
    if not research_result:
        return []

    payload = (
        research_result.get("synthesis_payload")
        or research_result.get("payload")
        or research_result
    )

    evidence = (
        payload.get("evidence")
        or payload.get("final_chunks")
        or research_result.get("evidence")
        or research_result.get("final_chunks")
        or []
    )

    ids = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        eid = item.get("evidence_id") or item.get("id")

        if eid:
            ids.append(str(eid).strip("[]"))

    return sorted(set(ids))


def execute_agent_plan(question, plan):
    # Validate user input and the Supervisor plan before tool execution.
    apply_pre_execution_guardrails(question, plan)

    route = plan["route"]

    data_result = None
    research_result = None

    started = time.perf_counter()

    if route in {"structured", "hybrid"}:
        data_result = run_data_agent(plan)

    if route in {"rag", "temporal_rag", "hybrid"}:
        research_result = run_research_agent(plan)

    synthesis = run_synthesis_agent(
        user_question=question,
        route=route,
        data_agent_result=data_result,
        research_agent_result=research_result,
    )

    allowed_eids = _extract_allowed_evidence_ids(
        research_result
    )

    post = apply_post_execution_guardrails(
        route=route,
        answer=synthesis["answer"],
        allowed_evidence_ids=allowed_eids,
    )

    total_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    return {
        "route": route,
        "data_result": data_result,
        "research_result": research_result,
        "synthesis_result": synthesis,
        "post_guardrail": post,
        "total_latency_ms": total_ms,
    }


SUPERVISOR_FN = run_supervisor_agent

assert callable(SUPERVISOR_FN)

print(
    "Supervisor callable resolved:",
    SUPERVISOR_FN.__name__,
)


# COMMAND ----------

# Supervisor contract smoke test.
supervisor_smoke_question = (
    "Show India's GDP growth from 2015 to 2025."
)

supervisor_smoke_plan = SUPERVISOR_FN(
    supervisor_smoke_question
)

assert isinstance(
    supervisor_smoke_plan,
    dict,
), (
    "Supervisor runtime contract failed. "
    f"Expected dict, got {type(supervisor_smoke_plan).__name__}"
)

assert supervisor_smoke_plan["route"] == "structured"
assert "NY.GDP.MKTP.KD.ZG" in supervisor_smoke_plan["indicators"]
assert supervisor_smoke_plan["observation_start_year"] == 2015
assert supervisor_smoke_plan["observation_end_year"] == 2025
assert supervisor_smoke_plan["report_years"] == []

print("Supervisor runtime contract PASSED.")

# COMMAND ----------

# One structured end-to-end smoke test.
# This must pass before the 40-case benchmark is allowed to run.
structured_smoke_execution = execute_agent_plan(
    supervisor_smoke_question,
    supervisor_smoke_plan,
)

structured_smoke_synthesis = structured_smoke_execution[
    "synthesis_result"
]

assert (
    structured_smoke_synthesis.get("status")
    == "success"
), (
    "Structured end-to-end smoke test failed: "
    f"{structured_smoke_synthesis}"
)

structured_smoke_answer = str(
    structured_smoke_synthesis.get("answer") or ""
)

assert (
    "did not find observation rows" not in structured_smoke_answer
), (
    "Structured synthesis did not render the Data Agent observations."
)

assert "2015" in structured_smoke_answer
assert "2025" in structured_smoke_answer

print("Structured synthesis observation rendering PASSED.")

print("Structured one-case end-to-end smoke test PASSED.")
print(
    json.dumps(
        {
            "route": structured_smoke_execution["route"],
            "status": structured_smoke_synthesis.get("status"),
            "answer": structured_smoke_synthesis.get("answer"),
            "total_latency_ms": structured_smoke_execution[
                "total_latency_ms"
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
)

# COMMAND ----------



# COMMAND ----------

# Targeted regression test 1:
# The previously failing temporal RAG case must return validated citations.
temporal_smoke_question = (
    "Compare the downside risks discussed in the January 2022 "
    "and January 2025 GEP reports."
)

temporal_smoke_plan = SUPERVISOR_FN(
    temporal_smoke_question
)

assert temporal_smoke_plan["route"] == "temporal_rag"
assert sorted(temporal_smoke_plan["report_years"]) == [2022, 2025]

temporal_smoke_execution = execute_agent_plan(
    temporal_smoke_question,
    temporal_smoke_plan,
)

temporal_smoke_research = temporal_smoke_execution[
    "research_result"
]

assert (
    temporal_smoke_research["citation_validation"]["citation_valid"]
    is True
)

print(
    "Temporal RAG citation regression test PASSED. "
    f"retry_applied={temporal_smoke_research.get('citation_retry_applied')}"
)

# COMMAND ----------

# Targeted regression test 2:
# Explicit regional aggregates must survive Supervisor normalization into
# the structured side of a hybrid plan.
hybrid_smoke_question = (
    "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 "
    "and explain the World Bank's January 2025 outlook for the two regions."
)

hybrid_smoke_plan = SUPERVISOR_FN(
    hybrid_smoke_question
)

assert hybrid_smoke_plan["route"] == "hybrid"

hybrid_entities = set(
    hybrid_smoke_plan.get("countries_or_entities") or []
)

assert "South Asia" in hybrid_entities
assert "Sub-Saharan Africa" in hybrid_entities

hybrid_smoke_execution = execute_agent_plan(
    hybrid_smoke_question,
    hybrid_smoke_plan,
)

assert (
    hybrid_smoke_execution["synthesis_result"].get("status")
    == "success"
)

print("Regional hybrid regression test PASSED.")

# COMMAND ----------

# Collect approved benchmark cases only after eval_df has been defined.
rows = [
    row.asDict(recursive=True)
    for row in eval_df.orderBy("eval_id").collect()
]

assert rows, "No evaluation rows were loaded."

run_id = uuid.uuid4().hex
results = []

print("Starting evaluation run:", run_id)
print("Cases:", len(rows))

for idx, case in enumerate(rows, start=1):
    eval_id = case["eval_id"]
    question = case["question"]

    print(
        f"[{idx}/{len(rows)}] "
        f"{eval_id} | expected={case['expected_route']}"
    )

    try:
        plan = SUPERVISOR_FN(question)

        if not isinstance(plan, dict):
            raise ValueError(
                "Supervisor must return a dictionary plan."
            )

        execution = execute_agent_plan(
            question,
            plan,
        )

        expected_route = case["expected_route"]
        actual_route = plan.get("route")

        route_correct = (
            actual_route == expected_route
        )

        expected_years = sorted(
            case.get("expected_report_years") or []
        )

        actual_years = sorted(
            plan.get("report_years") or []
        )

        report_years_correct = (
            expected_years == actual_years
        )

        synthesis = execution[
            "synthesis_result"
        ]

        results.append({
            "run_id": run_id,
            "eval_id": eval_id,
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "report_years_correct": report_years_correct,
            "status": synthesis.get("status"),
            "citation_valid": bool(
                synthesis
                .get("citation_validation", {})
                .get("citation_valid", True)
            ),
            "answer": synthesis.get("answer"),
            "total_latency_ms": execution[
                "total_latency_ms"
            ],
            "error": None,
            "evaluated_at": datetime.now(
                timezone.utc
            ),
        })

    except Exception as exc:
        results.append({
            "run_id": run_id,
            "eval_id": eval_id,
            "question": question,
            "expected_route": case[
                "expected_route"
            ],
            "actual_route": None,
            "route_correct": False,
            "report_years_correct": False,
            "status": "failed",
            "citation_valid": False,
            "answer": None,
            "total_latency_ms": None,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "evaluated_at": datetime.now(
                timezone.utc
            ),
        })

# COMMAND ----------

# ============================================================
# Persist agent evaluation results with an explicit Spark schema
# ============================================================

from pyspark.sql import types as T

assert results, "Evaluation produced no result rows."

# Explicit schema prevents CANNOT_DETERMINE_TYPE when a column
# contains only None values (for example, error or answer).
results_schema = T.StructType([
    T.StructField("run_id", T.StringType(), False),
    T.StructField("eval_id", T.StringType(), False),
    T.StructField("question", T.StringType(), False),
    T.StructField("expected_route", T.StringType(), False),
    T.StructField("actual_route", T.StringType(), True),
    T.StructField("route_correct", T.BooleanType(), False),
    T.StructField("report_years_correct", T.BooleanType(), False),
    T.StructField("status", T.StringType(), True),
    T.StructField("citation_valid", T.BooleanType(), False),
    T.StructField("answer", T.StringType(), True),
    T.StructField("total_latency_ms", T.DoubleType(), True),
    T.StructField("error", T.StringType(), True),
    T.StructField("evaluated_at", T.TimestampType(), False),
])

# Use the results already produced by the evaluation loop.
# Do not rerun the 40 cases.
results_df = spark.createDataFrame(
    results,
    schema=results_schema,
)

# Check the DataFrame before writing.
results_df.printSchema()

print(f"Evaluation rows ready: {results_df.count()}")

# Avoid duplicate rows if this cell is rerun for the same run_id.
# This deletes only rows belonging to the current evaluation run.
spark.sql(
    f"""
    DELETE FROM {RESULTS_TABLE}
    WHERE run_id = '{run_id}'
    """
)

# Persist the actual evaluation results.
(
    results_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable(RESULTS_TABLE)
)

display(
    results_df.orderBy("eval_id")
)

print("Evaluation results saved successfully.")

# COMMAND ----------

summary_df = (
    results_df
    .agg(
        F.count("*").alias("case_count"),
        F.avg(
            F.col("route_correct").cast("double")
        ).alias("route_accuracy"),
        F.avg(
            F.col("report_years_correct").cast("double")
        ).alias("report_year_accuracy"),
        F.avg(
            F.col("citation_valid").cast("double")
        ).alias("citation_validity"),
        F.avg(
            (F.col("status") == "success")
            .cast("double")
        ).alias("execution_success_rate"),
        F.avg(
            "total_latency_ms"
        ).alias("avg_total_latency_ms"),
    )
    .withColumn(
        "run_id",
        F.lit(run_id),
    )
    .withColumn(
        "evaluated_at",
        F.current_timestamp(),
    )
)

(
    summary_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable(SUMMARY_TABLE)
)

display(summary_df)

# Show failures separately for diagnosis.
failed_df = (
    results_df
    .filter(
        (F.col("status") != "success")
        | F.col("error").isNotNull()
    )
    .select(
        "eval_id",
        "expected_route",
        "actual_route",
        "error",
    )
    .orderBy("eval_id")
)

print("Failed cases:", failed_df.count())
display(failed_df)

print("Agent evaluation run_id:", run_id)
print("08_agent_evaluation_v3 COMPLETE")