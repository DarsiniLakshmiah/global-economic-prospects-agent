# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# Global Economic Prospects Intelligence Agent
# 02_build_retrieval_eval_dataset
#
# Updated version:
# - fixes the global-limit sampling bias
# - guarantees 16 candidates for each GEP edition, 2022-2026
# - keeps source provenance immutable
# - uses a fresh balanced-v2 LLM annotation checkpoint table
# - preserves deterministic screening, structured LLM annotation,
#   quality checks, human review controls, and final governed dataset logic


# COMMAND ----------

# # 02 — Build Retrieval Evaluation Dataset

# Purpose:
# - Build a source-grounded retrieval evaluation dataset from the existing GEP V3 child chunks.
# - Never invent document/page/chunk ground truth.
# - Filter obviously unsuitable evidence before annotation.
# - Keep source provenance fixed while questions are authored/reviewed.

# This notebook intentionally DOES NOT call an LLM yet.
# The next controlled step is question generation/validation from this governed review queue.

# Inputs:
# - worldbank_ai.silver.gep_child_chunks_enriched

# Outputs:
# - worldbank_ai.rag.retrieval_eval_candidate_pool
# - worldbank_ai.rag.retrieval_eval_review_queue
# - worldbank_ai.rag.retrieval_eval_dataset (only after approvals)

# Design rule:
# Evidence + provenance are selected first. Questions are added later.
# This prevents generated questions from deciding their own ground truth.

# COMMAND ----------

# ============================================================
# 1. Configuration
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

CATALOG = "worldbank_ai"
SILVER = "silver"
RAG = "rag"

SOURCE = f"{CATALOG}.{SILVER}.gep_child_chunks_enriched"
CANDIDATE_POOL = f"{CATALOG}.{RAG}.retrieval_eval_candidate_pool"
QUEUE = f"{CATALOG}.{RAG}.retrieval_eval_review_queue"
EVAL = f"{CATALOG}.{RAG}.retrieval_eval_dataset"

EXPECTED_YEARS = [2022, 2023, 2024, 2025, 2026]
TARGET_REVIEW_CASES = 80
FINAL_TARGET_CASES = 50

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{RAG}")

print("Configuration loaded.")
print(f"Source:         {SOURCE}")
print(f"Candidate pool: {CANDIDATE_POOL}")
print(f"Review queue:   {QUEUE}")
print(f"Final eval:     {EVAL}")

# COMMAND ----------

# ============================================================
# 2. Load and validate source
# ============================================================

source_df = spark.table(SOURCE)

required = {
    "chunk_id", "parent_chunk_id", "document_id", "report_year",
    "edition_status", "chapter", "section", "subsection", "region",
    "content_type", "page_start", "page_end", "chunk_text"
}

missing = sorted(required - set(source_df.columns))
assert not missing, f"Missing required columns: {missing}"

source_count = source_df.count()
years = [r["report_year"] for r in source_df.select("report_year").distinct().orderBy("report_year").collect()]

assert source_count == 2215, f"Expected 2,215 V3 child chunks, found {source_count:,}"
assert years == EXPECTED_YEARS, f"Expected years {EXPECTED_YEARS}, found {years}"
assert source_df.filter(F.col("chunk_id").isNull()).count() == 0
assert source_df.groupBy("chunk_id").count().filter(F.col("count") > 1).count() == 0
assert source_df.filter(F.col("chunk_text").isNull() | (F.length(F.trim("chunk_text")) == 0)).count() == 0

print(f"Source validation passed: {source_count:,} chunks; years={years}")

# COMMAND ----------

# ============================================================
# 3. Build evidence-quality features
# ============================================================
# These are conservative screening signals, not semantic truth labels.
# We keep the original source text unchanged.

text = F.col("chunk_text")
text_lower = F.lower(text)

# Bibliography/reference-like chunks are poor gold evidence for retrieval QA.
reference_pattern = (
    r"(?i)(\breferences\b|\bbibliography\b|doi:|https?://|"
    r"\bworld bank\.\s*20\d{2}\b|\bIMF\.\s*20\d{2}\b)"
)

# A useful evidence passage should normally contain some prose.
alpha_chars = F.length(F.regexp_replace(text, r"[^A-Za-z]", ""))
digit_chars = F.length(F.regexp_replace(text, r"[^0-9]", ""))
total_chars = F.length(text)

candidate_df = (
    source_df
    .withColumn("evidence_char_count", total_chars)
    .withColumn("alpha_char_count", alpha_chars)
    .withColumn("digit_char_count", digit_chars)
    .withColumn(
        "alpha_ratio",
        F.when(total_chars > 0, alpha_chars.cast("double") / total_chars).otherwise(F.lit(0.0))
    )
    .withColumn("looks_reference_heavy", text_lower.rlike(reference_pattern))
    .withColumn(
        "is_front_matter_flag",
        (F.lower(F.coalesce(F.col("content_type"), F.lit(""))) == "front_matter")
    )
    .withColumn(
        "is_too_short_for_gold",
        F.col("evidence_char_count") < 300
    )
    .withColumn(
        "has_basic_provenance",
        F.col("document_id").isNotNull()
        & F.col("report_year").isNotNull()
        & F.col("page_start").isNotNull()
        & F.col("page_end").isNotNull()
    )
    .withColumn(
        "baseline_candidate_eligible",
        (~F.col("is_front_matter_flag"))
        & (~F.col("is_too_short_for_gold"))
        & (~F.col("looks_reference_heavy"))
        & (F.col("alpha_ratio") >= F.lit(0.35))
        & F.col("has_basic_provenance")
    )
)

candidate_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(CANDIDATE_POOL)

print(f"Saved candidate pool: {CANDIDATE_POOL}")

display(
    candidate_df.groupBy("report_year", "baseline_candidate_eligible")
    .count()
    .orderBy("report_year", "baseline_candidate_eligible")
)

# COMMAND ----------

# ============================================================
# 4. Inspect exclusions before sampling
# ============================================================

display(
    candidate_df
    .filter(~F.col("baseline_candidate_eligible"))
    .select(
        "chunk_id", "report_year", "region", "section", "subsection",
        "content_type", "page_start", "page_end",
        "evidence_char_count", "alpha_ratio",
        "is_front_matter_flag", "is_too_short_for_gold",
        "looks_reference_heavy", "chunk_text"
    )
    .orderBy("report_year", "page_start")
    .limit(30)
)

# COMMAND ----------

# ============================================================
# 5. Create a balanced, diverse review-source sample
# ============================================================
# The earlier implementation used a global limit(80), which caused
# the queue to contain only 2022 and 2023.
#
# This version guarantees balanced representation:
#   2022 -> 16
#   2023 -> 16
#   2024 -> 16
#   2025 -> 16
#   2026 -> 16
#
# Within each year we still encourage diversity across region + section.
# Source evidence and provenance are never modified.

TARGET_PER_YEAR = TARGET_REVIEW_CASES // len(EXPECTED_YEARS)

assert TARGET_REVIEW_CASES % len(EXPECTED_YEARS) == 0, (
    "TARGET_REVIEW_CASES must divide evenly across EXPECTED_YEARS "
    "for this balanced benchmark design."
)

eligible = candidate_df.filter(F.col("baseline_candidate_eligible") == True)

sample_base = (
    eligible
    .withColumn(
        "_sample_region",
        F.when(
            F.length(F.trim(F.coalesce(F.col("region"), F.lit("")))) > 0,
            F.col("region")
        ).otherwise(F.lit("NO_REGION"))
    )
    .withColumn(
        "_sample_section",
        F.when(
            F.length(F.trim(F.coalesce(F.col("section"), F.lit("")))) > 0,
            F.col("section")
        ).otherwise(F.lit("NO_SECTION"))
    )
)

# First encourage structural diversity within each year.
diversity_window = (
    Window
    .partitionBy("report_year", "_sample_region", "_sample_section")
    .orderBy(F.sha2(F.col("chunk_id"), 256))
)

diverse_candidates = (
    sample_base
    .withColumn("_stratum_rank", F.row_number().over(diversity_window))
    .filter(F.col("_stratum_rank") <= 2)
)

# Then cap EACH year independently rather than applying a global limit.
year_window = (
    Window
    .partitionBy("report_year")
    .orderBy(
        F.sha2(
            F.concat_ws(
                "|",
                F.col("chunk_id"),
                F.col("_sample_region"),
                F.col("_sample_section")
            ),
            256
        )
    )
)

review_source = (
    diverse_candidates
    .withColumn("_year_rank", F.row_number().over(year_window))
    .filter(F.col("_year_rank") <= TARGET_PER_YEAR)
)

# Validate before the queue is written.
year_distribution = {
    r["report_year"]: r["count"]
    for r in review_source.groupBy("report_year").count().collect()
}

expected_distribution = {
    year: TARGET_PER_YEAR
    for year in EXPECTED_YEARS
}

assert year_distribution == expected_distribution, (
    f"Expected balanced distribution {expected_distribution}, "
    f"found {year_distribution}"
)

assert review_source.count() == TARGET_REVIEW_CASES, (
    f"Expected {TARGET_REVIEW_CASES} review candidates, "
    f"found {review_source.count()}"
)

print("Balanced sampling validation passed.")
display(
    review_source
    .groupBy("report_year")
    .count()
    .orderBy("report_year")
)

# COMMAND ----------

# ============================================================
# 6. Build review queue schema
# ============================================================
# IMPORTANT:
# - expected_* fields are suggestions inherited from the chunker.
# - reviewer_* fields are the human-validated values.
# - relevant_* provenance is fixed from the source chunk.

queue = (
    review_source
    .select(
        F.concat(F.lit("gep_eval_"), F.lpad(F.row_number().over(
            Window.orderBy("report_year", "document_id", "page_start", "chunk_id")
        ).cast("string"), 3, "0")).alias("eval_id"),

        # To be populated in the next annotation stage.
        F.lit(None).cast("string").alias("question"),
        F.lit(None).cast("string").alias("question_type"),
        F.lit(None).cast("string").alias("difficulty"),

        # Suggested metadata — NOT yet gold.
        F.col("report_year").alias("expected_report_year"),
        F.col("region").alias("suggested_region"),
        F.col("section").alias("suggested_section"),
        F.col("subsection").alias("suggested_subsection"),

        # Reviewer-confirmed metadata.
        F.lit(None).cast("string").alias("reviewer_region"),
        F.lit(None).cast("string").alias("reviewer_section"),
        F.lit(None).cast("string").alias("reviewer_subsection"),
        F.lit(None).cast("boolean").alias("metadata_matches_evidence"),

        # Immutable source provenance.
        F.col("document_id").alias("relevant_document_id"),
        F.col("page_start").alias("relevant_page_start"),
        F.col("page_end").alias("relevant_page_end"),
        F.col("chunk_id").alias("relevant_chunk_id"),
        F.col("parent_chunk_id").alias("relevant_parent_chunk_id"),
        F.col("content_type").alias("content_type"),
        F.col("chunk_text").alias("reference_evidence"),

        # Review controls.
        F.lit("PENDING").alias("review_status"),
        F.lit(False).alias("is_approved"),
        F.lit(None).cast("string").alias("rejection_reason"),
        F.lit(None).cast("string").alias("notes")
    )
)

assert queue.count() == TARGET_REVIEW_CASES

queue_year_distribution = {
    r["expected_report_year"]: r["count"]
    for r in queue.groupBy("expected_report_year").count().collect()
}
assert queue_year_distribution == expected_distribution, (
    f"Unexpected queue year distribution: {queue_year_distribution}"
)
assert queue.groupBy("eval_id").count().filter(F.col("count") > 1).count() == 0
assert queue.groupBy("relevant_chunk_id").count().filter(F.col("count") > 1).count() == 0

queue.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(QUEUE)

print(f"Saved review queue: {QUEUE}")
print(f"Review cases: {queue.count():,}")

# COMMAND ----------

# ============================================================
# 7. Inspect review queue
# ============================================================

display(
    spark.table(QUEUE)
    .select(
        "eval_id", "expected_report_year",
        "suggested_region", "suggested_section", "suggested_subsection",
        "content_type", "relevant_page_start", "relevant_page_end",
        "reference_evidence", "question", "review_status", "is_approved"
    )
    .orderBy("eval_id")
)

# COMMAND ----------

# ## STOP HERE ON THE FIRST RUN

# At this point the governed review queue exists.

# Do **not** approve all rows automatically.

# The next annotation step must:
# 1. Read `worldbank_ai.rag.retrieval_eval_review_queue`.
# 2. Check whether the suggested region/section/subsection agree with the evidence.
# 3. Generate or author a question that is answerable from the fixed `reference_evidence`.
# 4. Assign `question_type` and `difficulty`.
# 5. Reject unsuitable/mismatched evidence or correct the reviewer metadata.
# 6. Set `review_status='APPROVED'` and `is_approved=true` only after validation.

# Aim for about 50 approved cases from the ~80-case review queue.

# Recommended question mix:
# - global outlook / risks
# - regional outlook / risks / recent developments
# - policy questions
# - temporal/comparison cases (added carefully because they may require >1 evidence chunk)
# - table/figure cases only when the text representation is sufficient
# - paraphrased/harder retrieval questions

# The source document/chunk/page/evidence fields must not be rewritten.

# COMMAND ----------

# ============================================================
# 9. Configure Databricks LLM for evaluation annotation
# ============================================================
# This model is used ONLY to:
#   1. judge whether evidence is suitable for retrieval evaluation
#   2. check metadata against the actual evidence
#   3. generate a question FROM the evidence
#   4. classify question type and difficulty
#
# It is NOT allowed to:
#   - invent evidence
#   - change document/chunk/page provenance
#   - answer using outside knowledge
#
# We confirmed this endpoint exists in this workspace.

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

ANNOTATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"

print(f"Annotation model: {ANNOTATION_MODEL}")

# COMMAND ----------

# ============================================================
# 10. Test the Databricks LLM endpoint
# ============================================================
# IMPORTANT:
# In this Databricks SDK version, messages should be passed as
# ChatMessage objects rather than normal Python dictionaries.

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

test_response = w.serving_endpoints.query(
    name=ANNOTATION_MODEL,
    messages=[
        ChatMessage(
            role=ChatMessageRole.SYSTEM,
            content=(
                "You are a data quality assistant. "
                "Return concise answers and do not use outside information."
            )
        ),
        ChatMessage(
            role=ChatMessageRole.USER,
            content=(
                "Evidence: Global growth is projected to slow in 2022. "
                "Is this passage about an economic outlook? "
                "Answer only yes or no."
            )
        )
    ],
    temperature=0.0,
    max_tokens=10
)

print(test_response.choices[0].message.content)

# COMMAND ----------

# ============================================================
# 11. Structured annotation schema
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional, Literal


class EvalAnnotation(BaseModel):
    # Can this evidence support a meaningful retrieval question?
    evidence_usable: bool

    # Why it was accepted/rejected.
    evidence_quality_reason: str

    # Does the suggested structural metadata agree with the text?
    metadata_matches_evidence: bool

    # Correct metadata based ONLY on the evidence.
    # Null is allowed when the evidence is global or ambiguous.
    corrected_region: Optional[str] = None
    corrected_section: Optional[str] = None
    corrected_subsection: Optional[str] = None

    # Generated retrieval question.
    # Must be null if evidence_usable=False.
    question: Optional[str] = None

    # Controlled question categories.
    question_type: Optional[
        Literal[
            "global_outlook",
            "global_risk",
            "regional_outlook",
            "regional_risk",
            "recent_developments",
            "policy",
            "country_specific",
            "table_or_figure",
            "other"
        ]
    ] = None

    difficulty: Optional[
        Literal["easy", "medium", "hard"]
    ] = None

    # Critical validation signal.
    answerable_from_evidence: bool

    # Short explanation useful during human review.
    validation_reason: str


print("Annotation schema ready.")

# COMMAND ----------

# ============================================================
# 12. Build STRICT annotation prompt
# ============================================================
# Goal:
# Create consistent retrieval-evaluation annotations.
#
# IMPORTANT:
# The LLM may interpret the evidence, but it cannot change
# document/chunk/page provenance.

import json


def build_annotation_prompt(row):

    metadata = {
        "report_year": row["expected_report_year"],
        "suggested_region": row["suggested_region"],
        "suggested_section": row["suggested_section"],
        "suggested_subsection": row["suggested_subsection"],
        "content_type": row["content_type"],
        "page_start": row["relevant_page_start"],
        "page_end": row["relevant_page_end"],
    }

    return f"""
You are creating a retrieval evaluation benchmark for World Bank
Global Economic Prospects reports.

Evaluate ONE evidence passage using ONLY the supplied evidence.

============================================================
1. EVIDENCE QUALITY
============================================================

Set evidence_usable=false when the passage is mainly:

- glossary or abbreviation definitions
- bibliography or references
- isolated labels
- page numbers
- figure axis values
- disconnected numerical fragments
- source lists
- incomplete text without enough meaning
- text that cannot support a standalone question

A passage containing useful narrative plus figure/table content
CAN be usable.

============================================================
2. REGION CLASSIFICATION
============================================================

Determine region from the ACTUAL evidence.

Use ONLY one of these canonical values:

"East Asia and Pacific"
"Europe and Central Asia"
"Latin America and the Caribbean"
"Middle East and North Africa"
"South Asia"
"Sub-Saharan Africa"

Use null when:

- the evidence is global,
- multiple regions are discussed without one dominant region,
- or the region cannot be determined reliably.

IMPORTANT:

EAP = East Asia and Pacific
ECA = Europe and Central Asia
LAC = Latin America and the Caribbean
MENA or MNA = Middle East and North Africa
SAR or South Asia = South Asia
SSA = Sub-Saharan Africa

Do NOT include abbreviations in corrected_region.

Correct:
"Europe and Central Asia"

Incorrect:
"Europe and Central Asia (ECA)"

============================================================
3. SECTION CLASSIFICATION
============================================================

Use ONLY one of:

"Global Outlook"
"Recent developments"
"Outlook"
"Risks"
"Risks to the outlook"
"Policy challenges"
"Policy priorities"
"Policy implications"

Use null if none can be determined reliably.

Classify based on the PRIMARY purpose of the evidence.

Examples:

Historical/recent economic conditions:
    "Recent developments"

Forecasts and expected future growth:
    "Outlook"

Downside/upside uncertainties:
    "Risks" or "Risks to the outlook"

Policy recommendations/challenges:
    corresponding policy section

If a passage contains both historical developments and forecasts,
choose the section that best matches the QUESTION you generate.

============================================================
4. QUESTION TYPE
============================================================

Use EXACTLY one of:

global_outlook
global_risk
regional_outlook
regional_risk
recent_developments
policy
country_specific
table_or_figure
other

Rules:

global_outlook:
    Global/world economic outlook.
    No single World Bank region is the primary subject.

global_risk:
    Risks to the global/world economy.

regional_outlook:
    Outlook/forecast for ONE World Bank region.

regional_risk:
    Risks for ONE World Bank region.

recent_developments:
    Primarily asks about historical/recent economic developments.

policy:
    Primarily asks about policy actions, priorities or challenges.

country_specific:
    One specific country is the primary subject.

table_or_figure:
    The answer depends primarily on interpreting a table or figure.

other:
    Only when none of the above apply.

IMPORTANT:

If corrected_region is NOT null and the question asks about
future regional growth/outlook, question_type MUST be
"regional_outlook", NOT "global_outlook".

If corrected_region is NOT null and the question asks about
regional risks, question_type MUST be "regional_risk".

============================================================
5. QUESTION GENERATION
============================================================

Generate exactly ONE natural question.

The question must:

- be answerable entirely from the supplied evidence
- not require outside knowledge
- not reveal the answer
- not mention chunk IDs or page numbers
- sound like a realistic analyst/research question
- preferably paraphrase rather than copy a sentence verbatim
- be specific enough that retrieval quality can be measured

Avoid unnecessarily combining multiple independent questions.

============================================================
6. DIFFICULTY
============================================================

easy:
    answer is explicitly stated using similar wording

medium:
    requires connecting multiple statements or paraphrasing

hard:
    requires synthesizing several parts of the passage

============================================================
7. METADATA CONSISTENCY
============================================================

metadata_matches_evidence=true ONLY when the suggested region,
section and subsection are consistent with the actual evidence.

If an important suggested field is wrong, set it to false.

============================================================
8. ANSWERABILITY
============================================================

answerable_from_evidence=true ONLY if the generated question can
be answered completely from this evidence passage.

If evidence_usable=false:

- question must be null
- question_type must be null
- difficulty must be null
- answerable_from_evidence must be false

============================================================

SUGGESTED METADATA:

{json.dumps(metadata, indent=2)}

============================================================

REFERENCE EVIDENCE:

{row["reference_evidence"]}

============================================================

Return ONLY valid JSON.

Use exactly this schema:

{{
  "evidence_usable": true,
  "evidence_quality_reason": "...",
  "metadata_matches_evidence": false,
  "corrected_region": "Europe and Central Asia",
  "corrected_section": "Outlook",
  "corrected_subsection": "Outlook",
  "question": "...",
  "question_type": "regional_outlook",
  "difficulty": "medium",
  "answerable_from_evidence": true,
  "validation_reason": "..."
}}
"""


print("Strict annotation prompt ready.")

# COMMAND ----------

# ============================================================
# 13. Test annotation on ONE real evaluation candidate
# ============================================================

import json

review_queue = spark.table(QUEUE)

# Use the known suspicious example from our review queue.
test_row = (
    review_queue
    .filter(F.col("eval_id") == "gep_eval_019")
    .first()
)

assert test_row is not None, "Test evaluation row not found."

# Build the source-grounded prompt.
prompt = build_annotation_prompt(test_row)

# IMPORTANT:
# Use ChatMessage objects for this Databricks SDK version.
response = w.serving_endpoints.query(
    name=ANNOTATION_MODEL,
    messages=[
        ChatMessage(
            role=ChatMessageRole.SYSTEM,
            content=(
                "You are a strict retrieval-evaluation data annotator. "
                "Use only the supplied evidence. "
                "Return valid JSON only."
            )
        ),
        ChatMessage(
            role=ChatMessageRole.USER,
            content=prompt
        )
    ],
    temperature=0.0,
    max_tokens=700
)

raw_output = response.choices[0].message.content

print(raw_output)

# COMMAND ----------

# ============================================================
# Test a BAD evidence candidate
# ============================================================
# gep_eval_001 is mostly glossary / abbreviation material.
# A good annotation model should reject it rather than
# manufacture a retrieval question.

test_row = (
    review_queue
    .filter(F.col("eval_id") == "gep_eval_001")
    .first()
)

assert test_row is not None, "Test evaluation row not found."

prompt = build_annotation_prompt(test_row)

response = w.serving_endpoints.query(
    name=ANNOTATION_MODEL,
    messages=[
        ChatMessage(
            role=ChatMessageRole.SYSTEM,
            content=(
                "You are a strict retrieval-evaluation data annotator. "
                "Use only the supplied evidence. "
                "Return valid JSON only."
            )
        ),
        ChatMessage(
            role=ChatMessageRole.USER,
            content=prompt
        )
    ],
    temperature=0.0,
    max_tokens=700
)

raw_output = response.choices[0].message.content

print(raw_output)

# COMMAND ----------

# ============================================================
# 14. Validate structured LLM output
# ============================================================
# Never trust raw model output directly.
# Parse JSON and validate against our Pydantic schema.

clean_output = raw_output.strip()

# Some models may wrap JSON in markdown fences.
if clean_output.startswith("```"):
    clean_output = clean_output.replace("```json", "", 1)
    clean_output = clean_output.replace("```", "").strip()

parsed_json = json.loads(clean_output)

annotation = EvalAnnotation.model_validate(parsed_json)

print("Structured annotation validated successfully.")
print()
print(annotation.model_dump_json(indent=2))

# COMMAND ----------

# ============================================================
# 15. Batch annotation configuration
# ============================================================

import json
import time
import re
from datetime import datetime, timezone

from pyspark.sql import functions as F
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole


ANNOTATION_RESULTS_TABLE = (
    "worldbank_ai.rag.retrieval_eval_llm_annotations_balanced_v2"
)

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5

print("Batch annotation configuration")
print("--------------------------------")
print(f"Model:         {ANNOTATION_MODEL}")
print(f"Queue:         {QUEUE}")
print(f"Output table:  {ANNOTATION_RESULTS_TABLE}")
print(f"Max retries:   {MAX_RETRIES}")

# COMMAND ----------


# ============================================================
# 16. Robust structured-output parser
# ============================================================

def parse_annotation_response(raw_text: str) -> EvalAnnotation:
    """
    Convert the raw LLM response into a validated EvalAnnotation.

    Steps:
      1. Remove markdown fences if present.
      2. Locate the JSON object.
      3. Parse JSON.
      4. Validate using Pydantic.

    Raises an exception when validation fails.
    """

    if raw_text is None:
        raise ValueError("LLM returned no content.")

    text = raw_text.strip()

    # Remove common markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # Defensive extraction in case the model adds a small
    # amount of text around the JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No valid JSON object found in model response."
        )

    json_text = text[start:end + 1]

    parsed = json.loads(json_text)

    # Validate fields and allowed enum values.
    annotation = EvalAnnotation.model_validate(parsed)

    # Additional logical validation.
    if not annotation.evidence_usable:

        if annotation.question is not None:
            raise ValueError(
                "Rejected evidence unexpectedly contains a question."
            )

        if annotation.answerable_from_evidence:
            raise ValueError(
                "Rejected evidence cannot be marked answerable."
            )

    if annotation.evidence_usable:

        if not annotation.question:
            raise ValueError(
                "Usable evidence must contain a question."
            )

        if not annotation.question_type:
            raise ValueError(
                "Usable evidence must contain a question_type."
            )

        if not annotation.difficulty:
            raise ValueError(
                "Usable evidence must contain a difficulty."
            )

        if not annotation.answerable_from_evidence:
            raise ValueError(
                "Usable evidence must be answerable."
            )

    return annotation


print("Structured response parser ready.")

# COMMAND ----------

# ============================================================
# 17. Annotation function with retry handling
# ============================================================

def annotate_eval_row(row):
    """
    Annotate one review-queue row using the Databricks
    foundation model endpoint.

    IMPORTANT:
    This function NEVER changes:
      - eval_id
      - document_id
      - chunk_id
      - parent_chunk_id
      - page_start/page_end
      - reference_evidence

    Those remain source-controlled provenance.
    """

    prompt = build_annotation_prompt(row)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = w.serving_endpoints.query(
                name=ANNOTATION_MODEL,

                messages=[
                    ChatMessage(
                        role=ChatMessageRole.SYSTEM,
                        content=(
                            "You are a strict retrieval-evaluation "
                            "data annotator. "
                            "Use only the supplied evidence. "
                            "Return valid JSON only."
                        )
                    ),

                    ChatMessage(
                        role=ChatMessageRole.USER,
                        content=prompt
                    )
                ],

                temperature=0.0,
                max_tokens=700
            )

            raw_output = (
                response
                .choices[0]
                .message
                .content
            )

            annotation = parse_annotation_response(
                raw_output
            )

            return {
                "success": True,
                "annotation": annotation,
                "raw_output": raw_output,
                "attempts": attempt,
                "error": None
            }

        except Exception as exc:

            last_error = str(exc)

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed "
                f"for {row['eval_id']}: {last_error[:200]}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    # All attempts failed.
    return {
        "success": False,
        "annotation": None,
        "raw_output": None,
        "attempts": MAX_RETRIES,
        "error": last_error
    }


print("Annotation function ready.")

# COMMAND ----------

# ============================================================
# 18. Create annotation checkpoint table
# ============================================================

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {ANNOTATION_RESULTS_TABLE} (

    eval_id STRING,

    annotation_model STRING,

    evidence_usable BOOLEAN,
    evidence_quality_reason STRING,

    metadata_matches_evidence BOOLEAN,

    corrected_region STRING,
    corrected_section STRING,
    corrected_subsection STRING,

    generated_question STRING,
    question_type STRING,
    difficulty STRING,

    answerable_from_evidence BOOLEAN,
    validation_reason STRING,

    annotation_success BOOLEAN,
    annotation_attempts INT,
    annotation_error STRING,

    raw_model_output STRING,

    annotated_at TIMESTAMP

)
USING DELTA
""")

print(
    f"Checkpoint table ready: "
    f"{ANNOTATION_RESULTS_TABLE}"
)

# COMMAND ----------

# ============================================================
# 19. Determine which balanced-queue records still need annotation
# ============================================================

queue_df = spark.table(QUEUE)

successful_ids_df = (
    spark.table(ANNOTATION_RESULTS_TABLE)
    .filter(F.col("annotation_success") == True)
    .select("eval_id")
    .distinct()
)

current_successful_ids_df = (
    queue_df
    .select("eval_id")
    .join(successful_ids_df, on="eval_id", how="inner")
)

pending_df = (
    queue_df
    .join(current_successful_ids_df, on="eval_id", how="left_anti")
    .orderBy("eval_id")
)

total_queue = queue_df.count()
already_complete = current_successful_ids_df.count()
remaining = pending_df.count()

print(f"Total balanced queue records: {total_queue}")
print(f"Already annotated:            {already_complete}")
print(f"Remaining to process:         {remaining}")

assert total_queue == TARGET_REVIEW_CASES
assert already_complete + remaining == total_queue

# COMMAND ----------

# ============================================================
# 20. Run batch LLM annotation
# ============================================================
#
# Each completed record is checkpointed immediately to Delta.
#
# If execution stops:
#   rerun Cell 19
#   rerun Cell 20
#
# Successfully processed eval_ids will be skipped.
# ============================================================

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
    IntegerType,
    TimestampType,
)


annotation_schema = StructType([
    StructField("eval_id", StringType(), False),

    StructField("annotation_model", StringType(), False),

    StructField("evidence_usable", BooleanType(), True),
    StructField("evidence_quality_reason", StringType(), True),

    StructField("metadata_matches_evidence", BooleanType(), True),

    StructField("corrected_region", StringType(), True),
    StructField("corrected_section", StringType(), True),
    StructField("corrected_subsection", StringType(), True),

    StructField("generated_question", StringType(), True),
    StructField("question_type", StringType(), True),
    StructField("difficulty", StringType(), True),

    StructField("answerable_from_evidence", BooleanType(), True),
    StructField("validation_reason", StringType(), True),

    StructField("annotation_success", BooleanType(), False),
    StructField("annotation_attempts", IntegerType(), False),
    StructField("annotation_error", StringType(), True),

    StructField("raw_model_output", StringType(), True),

    StructField("annotated_at", TimestampType(), False),
])


# Small evaluation queue, so collecting these review records
# to the driver is appropriate.
pending_rows = pending_df.collect()

print(
    f"Starting annotation for "
    f"{len(pending_rows)} records..."
)


for index, row in enumerate(pending_rows, start=1):

    print(
        f"[{index}/{len(pending_rows)}] "
        f"{row['eval_id']}"
    )

    result = annotate_eval_row(row)

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    if result["success"]:

        a = result["annotation"]

        output_record = [(
            row["eval_id"],

            ANNOTATION_MODEL,

            a.evidence_usable,
            a.evidence_quality_reason,

            a.metadata_matches_evidence,

            a.corrected_region,
            a.corrected_section,
            a.corrected_subsection,

            a.question,
            a.question_type,
            a.difficulty,

            a.answerable_from_evidence,
            a.validation_reason,

            True,
            result["attempts"],
            None,

            result["raw_output"],

            now
        )]

    else:

        output_record = [(
            row["eval_id"],

            ANNOTATION_MODEL,

            None,
            None,

            None,

            None,
            None,
            None,

            None,
            None,
            None,

            None,
            None,

            False,
            result["attempts"],
            result["error"],

            result["raw_output"],

            now
        )]

    result_df = spark.createDataFrame(
        output_record,
        schema=annotation_schema
    )

    # MERGE makes this idempotent.
    result_df.createOrReplaceTempView(
        "current_eval_annotation"
    )

    spark.sql(f"""
        MERGE INTO {ANNOTATION_RESULTS_TABLE} AS target

        USING current_eval_annotation AS source

        ON target.eval_id = source.eval_id

        WHEN MATCHED THEN UPDATE SET *

        WHEN NOT MATCHED THEN INSERT *
    """)


print("Batch annotation run finished.")

# COMMAND ----------

# ============================================================
# 21. Batch annotation quality summary
# ============================================================

annotations_df = spark.table(
    ANNOTATION_RESULTS_TABLE
)


summary_df = (
    annotations_df
    .agg(

        F.count("*").alias(
            "total_annotations"
        ),

        F.sum(
            F.when(
                F.col("annotation_success"),
                1
            ).otherwise(0)
        ).alias(
            "successful"
        ),

        F.sum(
            F.when(
                ~F.col("annotation_success"),
                1
            ).otherwise(0)
        ).alias(
            "failed"
        ),

        F.sum(
            F.when(
                F.col("evidence_usable"),
                1
            ).otherwise(0)
        ).alias(
            "usable_evidence"
        ),

        F.sum(
            F.when(
                F.col("evidence_usable") == False,
                1
            ).otherwise(0)
        ).alias(
            "rejected_evidence"
        ),

        F.sum(
            F.when(
                F.col("metadata_matches_evidence") == False,
                1
            ).otherwise(0)
        ).alias(
            "metadata_mismatches"
        )
    )
)

display(summary_df)

# COMMAND ----------

# ============================================================
# Inspect generated evaluation questions
# ============================================================

display(
    annotations_df
    .filter(
        (F.col("annotation_success") == True) &
        (F.col("evidence_usable") == True) &
        (F.col("answerable_from_evidence") == True)
    )
    .select(
        "eval_id",
        "corrected_region",
        "corrected_section",
        "generated_question",
        "question_type",
        "difficulty",
        "metadata_matches_evidence"
    )
    .orderBy("eval_id")
)

# COMMAND ----------

# ============================================================
# Inspect rejected evidence
# ============================================================

display(
    annotations_df
    .filter(
        F.col("evidence_usable") == False
    )
    .select(
        "eval_id",
        "evidence_quality_reason",
        "validation_reason"
    )
    .orderBy("eval_id")
)

# COMMAND ----------

# ============================================================
# 22. Deterministic annotation quality checks
# ============================================================
# We do NOT call the LLM again here.
#
# Purpose:
#   - validate generated questions
#   - identify classification inconsistencies
#   - identify duplicate / near-duplicate questions
#   - identify suspicious metadata
#   - prepare candidates for final human review
#
# IMPORTANT:
# These checks do NOT change source provenance.

from pyspark.sql import functions as F
from pyspark.sql.window import Window


annotations_df = spark.table(ANNOTATION_RESULTS_TABLE)

usable_df = (
    annotations_df
    .filter(
        (F.col("annotation_success") == True) &
        (F.col("evidence_usable") == True) &
        (F.col("answerable_from_evidence") == True) &
        F.col("generated_question").isNotNull()
    )
)


# ------------------------------------------------------------
# Normalize question text for duplicate detection
# ------------------------------------------------------------

usable_df = usable_df.withColumn(
    "normalized_question",
    F.lower(
        F.regexp_replace(
            F.trim(F.col("generated_question")),
            r"[^a-zA-Z0-9 ]",
            ""
        )
    )
)


# ------------------------------------------------------------
# Exact duplicate detection
# ------------------------------------------------------------

duplicate_window = Window.partitionBy(
    "normalized_question"
)

usable_df = usable_df.withColumn(
    "exact_duplicate_count",
    F.count("*").over(duplicate_window)
)


# ------------------------------------------------------------
# Question-type consistency rules
# ------------------------------------------------------------
# These are conservative checks.
# They FLAG questionable rows rather than automatically
# rewriting the LLM output.

usable_df = usable_df.withColumn(

    "type_consistency_issue",

    F.when(

        # Policy sections should normally produce policy questions.
        F.col("corrected_section").isin(
            "Policy challenges",
            "Policy priorities",
            "Policy implications"
        )
        &
        (F.col("question_type") != "policy"),

        F.lit("POLICY_SECTION_TYPE_MISMATCH")
    )

    .when(

        # Explicit regional outlook should normally be regional_outlook.
        (
            F.col("corrected_region").isNotNull()
            &
            (F.col("corrected_section") == "Outlook")
            &
            ~F.col("question_type").isin(
                "regional_outlook",
                "country_specific",
                "table_or_figure"
            )
        ),

        F.lit("REGIONAL_OUTLOOK_TYPE_MISMATCH")
    )

    .when(

        # Explicit regional risks should normally be regional_risk.
        (
            F.col("corrected_region").isNotNull()
            &
            F.col("corrected_section").isin(
                "Risks",
                "Risks to the outlook"
            )
            &
            ~F.col("question_type").isin(
                "regional_risk",
                "country_specific",
                "table_or_figure"
            )
        ),

        F.lit("REGIONAL_RISK_TYPE_MISMATCH")
    )

    .when(

        # Recent-development evidence classified as a risk/outlook
        # deserves human inspection.
        (
            (F.col("corrected_section") == "Recent developments")
            &
            F.col("question_type").isin(
                "regional_outlook",
                "regional_risk",
                "global_outlook",
                "global_risk"
            )
        ),

        F.lit("RECENT_DEVELOPMENT_TYPE_MISMATCH")
    )

    .otherwise(F.lit(None))
)


# ------------------------------------------------------------
# Additional quality flags
# ------------------------------------------------------------

usable_df = usable_df.withColumn(
    "question_too_short",
    F.length(F.col("generated_question")) < 25
)

usable_df = usable_df.withColumn(
    "question_too_long",
    F.length(F.col("generated_question")) > 300
)

usable_df = usable_df.withColumn(
    "missing_question_mark",
    ~F.trim(F.col("generated_question")).endswith("?")
)

usable_df = usable_df.withColumn(
    "exact_duplicate",
    F.col("exact_duplicate_count") > 1
)


# ------------------------------------------------------------
# Overall deterministic review flag
# ------------------------------------------------------------

usable_df = usable_df.withColumn(

    "needs_human_review",

    F.col("type_consistency_issue").isNotNull()
    |
    F.col("question_too_short")
    |
    F.col("question_too_long")
    |
    F.col("missing_question_mark")
    |
    F.col("exact_duplicate")
)


display(
    usable_df.select(
        "eval_id",
        "corrected_region",
        "corrected_section",
        "generated_question",
        "question_type",
        "difficulty",
        "type_consistency_issue",
        "exact_duplicate",
        "needs_human_review"
    )
    .orderBy(
        F.desc("needs_human_review"),
        "eval_id"
    )
)

# COMMAND ----------

# ============================================================
# 23. Evaluation benchmark coverage diagnostics
# ============================================================

print("YEAR DISTRIBUTION")
display(
    spark.table(QUEUE)
    .join(
        usable_df.select("eval_id"),
        "eval_id",
        "inner"
    )
    .groupBy("expected_report_year")
    .count()
    .orderBy("expected_report_year")
)


print("QUESTION TYPE DISTRIBUTION")
display(
    usable_df
    .groupBy("question_type")
    .count()
    .orderBy(F.desc("count"))
)


print("DIFFICULTY DISTRIBUTION")
display(
    usable_df
    .groupBy("difficulty")
    .count()
    .orderBy("difficulty")
)


print("REGION DISTRIBUTION")
display(
    usable_df
    .groupBy(
        F.coalesce(
            F.col("corrected_region"),
            F.lit("GLOBAL / NON-REGIONAL")
        ).alias("region")
    )
    .count()
    .orderBy(F.desc("count"))
)


print("SECTION DISTRIBUTION")
display(
    usable_df
    .groupBy("corrected_section")
    .count()
    .orderBy(F.desc("count"))
)

# The source queue itself must remain balanced even though some evidence
# may later be rejected by the annotation model.
queue_year_check = {
    r["expected_report_year"]: r["count"]
    for r in spark.table(QUEUE).groupBy("expected_report_year").count().collect()
}
assert queue_year_check == expected_distribution, (
    f"Balanced queue changed unexpectedly: {queue_year_check}"
)
print("Balanced queue integrity check passed.")

# COMMAND ----------

# ============================================================
# 24. Records requiring review
# ============================================================

review_needed_df = (
    usable_df
    .filter(F.col("needs_human_review") == True)
    .select(
        "eval_id",
        "corrected_region",
        "corrected_section",
        "generated_question",
        "question_type",
        "difficulty",
        "type_consistency_issue",
        "exact_duplicate"
    )
    .orderBy("eval_id")
)

print(
    f"Usable questions: {usable_df.count()}"
)

print(
    f"Flagged for review: {review_needed_df.count()}"
)

display(review_needed_df)

# COMMAND ----------

# ============================================================
# 8. Annotation progress check
# ============================================================
# Safe to run any time. On the first run everything should be PENDING.

annotated = spark.table(QUEUE)

display(
    annotated
    .groupBy("review_status", "is_approved")
    .count()
    .orderBy("review_status", "is_approved")
)

approved_count = annotated.filter(
    (F.col("is_approved") == True)
    & (F.col("review_status") == "APPROVED")
    & F.col("question").isNotNull()
    & (F.length(F.trim(F.col("question"))) > 0)
).count()

print(f"Approved cases currently available: {approved_count}")
print(f"Target final evaluation cases: approximately {FINAL_TARGET_CASES}")

# COMMAND ----------

# ============================================================
# 9. Build final governed evaluation dataset
# ============================================================
# Run this ONLY after annotation/review has produced approved rows.

annotated = spark.table(QUEUE)

approved = annotated.filter(
    (F.col("is_approved") == True)
    & (F.col("review_status") == "APPROVED")
    & F.col("question").isNotNull()
    & (F.length(F.trim(F.col("question"))) > 0)
    & F.col("metadata_matches_evidence").isNotNull()
)

approved_count = approved.count()
print(f"Approved evaluation cases: {approved_count}")

if approved_count == 0:
    print(
        "No approved cases yet. This is expected on the first run. "
        "Complete the annotation/review stage, then rerun this cell."
    )
else:
    # Use reviewer-confirmed metadata when supplied.
    final_eval = (
        approved
        .withColumn(
            "expected_region",
            F.coalesce(F.col("reviewer_region"), F.col("suggested_region"))
        )
        .withColumn(
            "expected_section",
            F.coalesce(F.col("reviewer_section"), F.col("suggested_section"))
        )
        .withColumn(
            "expected_subsection",
            F.coalesce(F.col("reviewer_subsection"), F.col("suggested_subsection"))
        )
        .select(
            "eval_id",
            "question",
            "question_type",
            "difficulty",
            "expected_report_year",
            "expected_region",
            "expected_section",
            "expected_subsection",
            "relevant_document_id",
            "relevant_page_start",
            "relevant_page_end",
            "relevant_chunk_id",
            "relevant_parent_chunk_id",
            "content_type",
            "reference_evidence",
            "metadata_matches_evidence",
            "notes"
        )
    )

    # Final integrity checks.
    assert final_eval.groupBy("eval_id").count().filter(F.col("count") > 1).count() == 0
    assert final_eval.filter(F.col("reference_evidence").isNull()).count() == 0
    assert final_eval.filter(F.col("relevant_chunk_id").isNull()).count() == 0

    final_eval.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(EVAL)

    print(f"Saved final retrieval evaluation dataset: {EVAL}")

    display(
        final_eval
        .groupBy("question_type", "difficulty")
        .count()
        .orderBy("question_type", "difficulty")
    )

# COMMAND ----------

# ============================================================
# 10. Final dataset diagnostics
# ============================================================
# This cell is safe even if the final table has not been created yet.

if spark.catalog.tableExists(EVAL):
    final_df = spark.table(EVAL)

    print(f"Final eval rows: {final_df.count():,}")

    display(
        final_df.groupBy("expected_report_year")
        .count()
        .orderBy("expected_report_year")
    )

    display(
        final_df.select(
            "eval_id", "question", "question_type", "difficulty",
            "expected_report_year", "expected_region",
            "expected_section", "relevant_page_start",
            "relevant_page_end", "relevant_chunk_id"
        ).orderBy("eval_id")
    )
else: 
    print("Final evaluation table does not exist yet — expected before annotation is complete.")

# COMMAND ----------

# ## MLflow Evaluation Dataset

# Do not sync to MLflow from this notebook until the final governed Delta
# evaluation dataset has been reviewed and is stable.

# We will perform the MLflow sync in the retrieval-evaluation stage so that
# the benchmark version, retrieval configuration, metrics, and experiment
# run are recorded together.


# COMMAND ----------

# ============================================================
# 25. Build clean final-benchmark candidate set
# ============================================================
# We are continuing from persisted Delta tables.
#
# NO LLM calls happen here.
#
# Goal:
#   80 balanced source candidates
#       ↓
#   73 usable LLM annotations
#       ↓
#   remove rejected / failed / flagged / duplicate cases
#       ↓
#   clean candidates for final benchmark selection
#
# IMPORTANT:
# Source provenance remains unchanged.
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window


queue_df = spark.table(QUEUE)

annotations_df = spark.table(
    ANNOTATION_RESULTS_TABLE
)


# ------------------------------------------------------------
# Join annotations back to immutable source provenance
# ------------------------------------------------------------

benchmark_candidates = (
    queue_df.alias("q")
    .join(
        annotations_df.alias("a"),
        on="eval_id",
        how="inner"
    )
    .select(

        F.col("eval_id"),

        # Generated evaluation question
        F.col("a.generated_question")
        .alias("question"),

        F.col("a.question_type")
        .alias("question_type"),

        F.col("a.difficulty")
        .alias("difficulty"),

        # Report edition
        F.col("q.expected_report_year")
        .alias("expected_report_year"),

        # LLM-corrected metadata
        F.col("a.corrected_region")
        .alias("expected_region"),

        F.col("a.corrected_section")
        .alias("expected_section"),

        F.col("a.corrected_subsection")
        .alias("expected_subsection"),

        # ----------------------------------------------------
        # IMMUTABLE SOURCE PROVENANCE
        # ----------------------------------------------------

        F.col("q.relevant_document_id"),
        F.col("q.relevant_page_start"),
        F.col("q.relevant_page_end"),
        F.col("q.relevant_chunk_id"),
        F.col("q.relevant_parent_chunk_id"),

        F.col("q.content_type"),
        F.col("q.reference_evidence"),

        # Annotation quality information
        F.col("a.evidence_usable"),
        F.col("a.answerable_from_evidence"),
        F.col("a.metadata_matches_evidence"),
        F.col("a.annotation_success"),

        F.col("a.evidence_quality_reason"),
        F.col("a.validation_reason")
    )
)


# ------------------------------------------------------------
# Keep only successful + usable + answerable records
# ------------------------------------------------------------

usable_candidates = (
    benchmark_candidates
    .filter(
        (F.col("annotation_success") == True)
        &
        (F.col("evidence_usable") == True)
        &
        (F.col("answerable_from_evidence") == True)
        &
        F.col("question").isNotNull()
    )
)


print(
    f"Usable benchmark candidates: "
    f"{usable_candidates.count()}"
)

display(
    usable_candidates
    .groupBy("expected_report_year")
    .count()
    .orderBy("expected_report_year")
)

# COMMAND ----------

# ============================================================
# 26. Final deterministic quality gate
# ============================================================
# We intentionally exclude questionable records from automatic
# approval rather than silently rewriting their labels.
# ============================================================


quality_df = (
    usable_candidates

    # Normalize question for exact duplicate detection
    .withColumn(
        "_normalized_question",
        F.lower(
            F.regexp_replace(
                F.trim(F.col("question")),
                r"[^a-zA-Z0-9 ]",
                ""
            )
        )
    )
)


# ------------------------------------------------------------
# Exact duplicate count
# ------------------------------------------------------------

duplicate_window = Window.partitionBy(
    "_normalized_question"
)

quality_df = quality_df.withColumn(
    "_duplicate_count",
    F.count("*").over(duplicate_window)
)


# ------------------------------------------------------------
# Question-type consistency checks
# ------------------------------------------------------------

quality_df = quality_df.withColumn(

    "_type_issue",

    # Policy evidence should normally produce policy questions.
    F.when(
        F.col("expected_section").isin(
            "Policy challenges",
            "Policy priorities",
            "Policy implications"
        )
        &
        (F.col("question_type") != "policy"),

        F.lit("POLICY_SECTION_TYPE_MISMATCH")
    )

    # Regional outlook
    .when(
        F.col("expected_region").isNotNull()
        &
        (F.col("expected_section") == "Outlook")
        &
        ~F.col("question_type").isin(
            "regional_outlook",
            "country_specific",
            "table_or_figure"
        ),

        F.lit("REGIONAL_OUTLOOK_TYPE_MISMATCH")
    )

    # Regional risks
    .when(
        F.col("expected_region").isNotNull()
        &
        F.col("expected_section").isin(
            "Risks",
            "Risks to the outlook"
        )
        &
        ~F.col("question_type").isin(
            "regional_risk",
            "country_specific",
            "table_or_figure"
        ),

        F.lit("REGIONAL_RISK_TYPE_MISMATCH")
    )

    # Recent developments should not silently become
    # risk/outlook questions.
    .when(
        (F.col("expected_section") == "Recent developments")
        &
        F.col("question_type").isin(
            "regional_outlook",
            "regional_risk",
            "global_outlook",
            "global_risk"
        ),

        F.lit("RECENT_DEVELOPMENT_TYPE_MISMATCH")
    )

    .otherwise(F.lit(None))
)


# ------------------------------------------------------------
# Additional question-quality checks
# ------------------------------------------------------------

quality_df = (
    quality_df

    .withColumn(
        "_exact_duplicate",
        F.col("_duplicate_count") > 1
    )

    .withColumn(
        "_question_too_short",
        F.length(F.col("question")) < 25
    )

    .withColumn(
        "_question_too_long",
        F.length(F.col("question")) > 300
    )

    .withColumn(
        "_missing_question_mark",
        ~F.trim(F.col("question")).endswith("?")
    )
)


quality_df = quality_df.withColumn(

    "_needs_review",

    F.col("_type_issue").isNotNull()
    |
    F.col("_exact_duplicate")
    |
    F.col("_question_too_short")
    |
    F.col("_question_too_long")
    |
    F.col("_missing_question_mark")
)


clean_candidates = (
    quality_df
    .filter(F.col("_needs_review") == False)
)


flagged_candidates = (
    quality_df
    .filter(F.col("_needs_review") == True)
)


print(
    f"Usable candidates:  {quality_df.count()}"
)

print(
    f"Clean candidates:   {clean_candidates.count()}"
)

print(
    f"Flagged candidates: {flagged_candidates.count()}"
)


display(
    flagged_candidates
    .select(
        "eval_id",
        "expected_report_year",
        "expected_region",
        "expected_section",
        "question",
        "question_type",
        "difficulty",
        "_type_issue",
        "_exact_duplicate"
    )
    .orderBy("eval_id")
)

# COMMAND ----------

# ============================================================
# 27. Select balanced final benchmark
# ============================================================
# Target:
#
#   2022 -> 10
#   2023 -> 10
#   2024 -> 10
#   2025 -> 10
#   2026 -> 10
#
# Total = 50
#
# Selection is deterministic.
# ============================================================

FINAL_PER_YEAR = 10
FINAL_TARGET = 50


# ------------------------------------------------------------
# Create a deterministic diversity key
# ------------------------------------------------------------
# The ordering incorporates:
#   region
#   section
#   question type
#   difficulty
#   eval_id
#
# This is preferable to simply taking the first 10 rows.
# ------------------------------------------------------------

selection_window = (
    Window
    .partitionBy("expected_report_year")
    .orderBy(

        F.sha2(
            F.concat_ws(
                "|",

                F.coalesce(
                    F.col("expected_region"),
                    F.lit("GLOBAL")
                ),

                F.coalesce(
                    F.col("expected_section"),
                    F.lit("NO_SECTION")
                ),

                F.coalesce(
                    F.col("question_type"),
                    F.lit("NO_TYPE")
                ),

                F.coalesce(
                    F.col("difficulty"),
                    F.lit("NO_DIFFICULTY")
                ),

                F.col("eval_id")
            ),
            256
        )
    )
)


ranked_candidates = (
    clean_candidates
    .withColumn(
        "_year_selection_rank",
        F.row_number().over(selection_window)
    )
)


final_selected = (
    ranked_candidates
    .filter(
        F.col("_year_selection_rank")
        <= FINAL_PER_YEAR
    )
)


# ------------------------------------------------------------
# Validate availability
# ------------------------------------------------------------

final_count = final_selected.count()

print(
    f"Selected benchmark cases: {final_count}"
)


display(
    final_selected
    .groupBy("expected_report_year")
    .count()
    .orderBy("expected_report_year")
)

# COMMAND ----------

# ============================================================
# 28. Final benchmark diversity checks
# ============================================================

print("QUESTION TYPES")
display(
    final_selected
    .groupBy("question_type")
    .count()
    .orderBy(F.desc("count"))
)


print("DIFFICULTY")
display(
    final_selected
    .groupBy("difficulty")
    .count()
    .orderBy("difficulty")
)


print("REGIONS")
display(
    final_selected
    .groupBy(
        F.coalesce(
            F.col("expected_region"),
            F.lit("GLOBAL / NON-REGIONAL")
        ).alias("region")
    )
    .count()
    .orderBy(F.desc("count"))
)


print("SECTIONS")
display(
    final_selected
    .groupBy(
        F.coalesce(
            F.col("expected_section"),
            F.lit("NO SECTION")
        ).alias("section")
    )
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# ============================================================
# 29. Final benchmark hard validation
# ============================================================

assert final_selected.count() == FINAL_TARGET, (
    f"Expected {FINAL_TARGET} cases, "
    f"found {final_selected.count()}"
)


# Exactly 10 from each year
year_counts = {
    row["expected_report_year"]: row["count"]
    for row in (
        final_selected
        .groupBy("expected_report_year")
        .count()
        .collect()
    )
}

expected_year_counts = {
    2022: 10,
    2023: 10,
    2024: 10,
    2025: 10,
    2026: 10
}

assert year_counts == expected_year_counts, (
    f"Year imbalance detected: {year_counts}"
)


# No duplicate evaluation IDs
assert (
    final_selected
    .select("eval_id")
    .distinct()
    .count()
    == FINAL_TARGET
)


# No duplicate questions
assert (
    final_selected
    .select("_normalized_question")
    .distinct()
    .count()
    == FINAL_TARGET
)


# No duplicate source chunks
assert (
    final_selected
    .select("relevant_chunk_id")
    .distinct()
    .count()
    == FINAL_TARGET
)


# Every question has source evidence
assert (
    final_selected
    .filter(
        F.col("reference_evidence").isNull()
    )
    .count()
    == 0
)


# Every question has document provenance
assert (
    final_selected
    .filter(
        F.col("relevant_document_id").isNull()
    )
    .count()
    == 0
)


# Every question has page provenance
assert (
    final_selected
    .filter(
        F.col("relevant_page_start").isNull()
        |
        F.col("relevant_page_end").isNull()
    )
    .count()
    == 0
)


# Nothing selected should still require review
assert (
    final_selected
    .filter(F.col("_needs_review") == True)
    .count()
    == 0
)


print("All final benchmark validation checks passed.")

# COMMAND ----------

# ============================================================
# 30. Persist final retrieval evaluation dataset
# ============================================================
# This becomes the benchmark used for:
#
#   V1 fixed-size retrieval
#   V2 recursive retrieval
#   V3 structure-aware retrieval
#
# IMPORTANT:
# relevant_chunk_id is provenance from the V3 source evidence.
#
# We must NOT evaluate V1/V2 by requiring this same chunk ID.
#
# Cross-strategy retrieval scoring should use:
#   document match
#   + page overlap
#   + reference evidence
# ============================================================

from pyspark.sql import functions as F


# ------------------------------------------------------------
# Define final output table explicitly
# ------------------------------------------------------------
# We define it here so this cell can run independently without
# depending on variables created in earlier notebook cells.

EVAL_DATASET = (
    "worldbank_ai.rag.retrieval_eval_dataset"
)


# ------------------------------------------------------------
# Build final approved evaluation dataset
# ------------------------------------------------------------

final_eval_dataset = (
    final_selected
    .select(

        # Evaluation identity
        "eval_id",

        # Question information
        "question",
        "question_type",
        "difficulty",

        # Expected retrieval metadata
        "expected_report_year",
        "expected_region",
        "expected_section",
        "expected_subsection",

        # Ground-truth document/page provenance
        "relevant_document_id",
        "relevant_page_start",
        "relevant_page_end",

        # V3 source provenance
        # Do NOT use this chunk ID as universal ground truth
        # when comparing V1/V2/V3 chunking strategies.
        "relevant_chunk_id",
        "relevant_parent_chunk_id",

        # Evidence information
        "content_type",
        "reference_evidence",

        # Approval metadata
        F.lit(True)
        .alias("is_approved"),

        F.current_timestamp()
        .alias("approved_at"),

        # Dataset version
        F.lit("balanced_v1")
        .alias("evaluation_dataset_version")
    )
)


# ------------------------------------------------------------
# Pre-write validation
# ------------------------------------------------------------

final_count = final_eval_dataset.count()

assert final_count == 50, (
    f"Expected 50 final evaluation cases, "
    f"found {final_count}"
)

assert (
    final_eval_dataset
    .select("eval_id")
    .distinct()
    .count()
    == 50
), "Duplicate eval_id detected."

assert (
    final_eval_dataset
    .select("question")
    .distinct()
    .count()
    == 50
), "Duplicate question detected."

assert (
    final_eval_dataset
    .filter(F.col("question").isNull())
    .count()
    == 0
), "Null question detected."

assert (
    final_eval_dataset
    .filter(F.col("reference_evidence").isNull())
    .count()
    == 0
), "Null reference evidence detected."

assert (
    final_eval_dataset
    .filter(F.col("relevant_document_id").isNull())
    .count()
    == 0
), "Missing document provenance detected."

assert (
    final_eval_dataset
    .filter(
        F.col("relevant_page_start").isNull()
        |
        F.col("relevant_page_end").isNull()
    )
    .count()
    == 0
), "Missing page provenance detected."


print(
    f"Pre-write validation passed: "
    f"{final_count} benchmark cases."
)


# ------------------------------------------------------------
# Persist as governed Delta table
# ------------------------------------------------------------

(
    final_eval_dataset
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(EVAL_DATASET)
)


print(
    f"Final evaluation dataset written successfully: "
    f"{EVAL_DATASET}"
)

# COMMAND ----------

# ============================================================
# 31. Final persisted benchmark validation
# ============================================================

persisted_eval = spark.table(
    EVAL_DATASET
)


print(
    f"Final benchmark rows: "
    f"{persisted_eval.count()}"
)


print("\nYEAR COVERAGE")

display(
    persisted_eval
    .groupBy("expected_report_year")
    .count()
    .orderBy("expected_report_year")
)


print("\nQUESTION TYPES")

display(
    persisted_eval
    .groupBy("question_type")
    .count()
    .orderBy(F.desc("count"))
)


print("\nDIFFICULTY")

display(
    persisted_eval
    .groupBy("difficulty")
    .count()
    .orderBy("difficulty")
)


print("\nSAMPLE BENCHMARK QUESTIONS")

display(
    persisted_eval
    .select(
        "eval_id",
        "question",
        "question_type",
        "difficulty",
        "expected_report_year",
        "expected_region",
        "expected_section",
        "relevant_page_start",
        "relevant_page_end"
    )
    .orderBy(
        "expected_report_year",
        "eval_id"
    )
)


# ------------------------------------------------------------
# Final read-back assertions
# ------------------------------------------------------------

assert persisted_eval.count() == 50

assert (
    persisted_eval
    .select("eval_id")
    .distinct()
    .count()
    == 50
)

assert (
    persisted_eval
    .select("question")
    .distinct()
    .count()
    == 50
)

assert (
    persisted_eval
    .select("relevant_chunk_id")
    .distinct()
    .count()
    == 50
)


print(
    "\nRetrieval evaluation benchmark is ready."
)

# COMMAND ----------

