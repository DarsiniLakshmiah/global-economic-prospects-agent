# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05_tools_agents / 05_research_agent
# MAGIC
# MAGIC Agent layer for governed Global Economic Prospects (GEP) research.
# MAGIC
# MAGIC This notebook sits **above** the locked Research/RAG capability from Notebook 02.
# MAGIC It consumes a validated Supervisor plan and executes the qualitative research portion.
# MAGIC
# MAGIC ### Responsibilities
# MAGIC - validate Supervisor research plans
# MAGIC - support `rag`, `temporal_rag`, and the research side of `hybrid`
# MAGIC - enforce GEP report years 2022–2026
# MAGIC - use the locked structure-aware AI Search index
# MAGIC - use HYBRID retrieval with report-year filtering
# MAGIC - build compact evidence context
# MAGIC - generate a grounded research answer with `[E#]` citations
# MAGIC - validate citations and return provenance for the Synthesis Agent
# MAGIC
# MAGIC ### Non-responsibilities
# MAGIC - no structured macroeconomic lookup
# MAGIC - no arbitrary index selection
# MAGIC - no unrestricted filters
# MAGIC - no final cross-source synthesis
# MAGIC - no default LLM reranking
# MAGIC
# MAGIC The baseline remains:
# MAGIC structure-aware chunks → Qwen3 managed embeddings → Databricks AI Search →
# MAGIC HYBRID retrieval → report-year filter → top evidence → context →
# MAGIC Llama generation → citation validation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01. Install only the required Databricks clients
# MAGIC
# MAGIC If your compute already has these versions, this cell should be quick.
# MAGIC We restart Python once so the installed SDK is loaded consistently.

# COMMAND ----------

# MAGIC %pip install -q databricks-ai-search databricks-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02. Imports and frozen production configuration

# COMMAND ----------

from typing import Any, Dict, List, Optional
from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI
import json
import math
import re
import time

AI_SEARCH_ENDPOINT = "worldbank-gep-ai-search"
INDEX_NAME = "worldbank_ai.rag.gep_structure_v1_qwen3_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

VALID_REPORT_YEARS = {2022, 2023, 2024, 2025, 2026}

RETRIEVAL_TOP_K = 6
MAX_CONTEXT_TOKENS = 6000
CHARS_PER_TOKEN = 4.0
MAX_TOKENS_PER_EVIDENCE = 1400
NEAR_DUPLICATE_JACCARD = 0.82
MAX_ANSWER_TOKENS = 1200

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

print("Research Agent configuration loaded.")
print("AI Search endpoint:", AI_SEARCH_ENDPOINT)
print("Index:", INDEX_NAME)
print("LLM endpoint:", LLM_ENDPOINT)
print("Valid GEP editions:", sorted(VALID_REPORT_YEARS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03. Connect to the locked AI Search index and serving endpoint
# MAGIC
# MAGIC Important: this environment uses `databricks.ai_search.client.AISearchClient`.
# MAGIC We do not use the legacy Vector Search client.

# COMMAND ----------

ai_search_client = AISearchClient()

index = ai_search_client.get_index(
    endpoint_name=AI_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
)

llm_client = DatabricksOpenAI()

print("AI Search client initialized.")
print("Databricks model-serving client initialized.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04. Model-serving smoke test

# COMMAND ----------

model_test = llm_client.chat.completions.create(
    model=LLM_ENDPOINT,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly MODEL_OK",
        }
    ],
    temperature=0,
    max_tokens=10,
)

model_test_text = model_test.choices[0].message.content.strip()

print("Model test:", model_test_text)

assert "MODEL_OK" in model_test_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05. JSON-safe helpers

# COMMAND ----------

def json_safe(value: Any) -> Any:
    """Convert nested Python values into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    return str(value)


def normalize_years(
    report_years: Optional[List[int]],
) -> List[int]:
    """Validate and de-duplicate GEP edition years."""
    if not report_years:
        return []

    years = []

    for year in report_years:
        year = int(year)

        if year not in VALID_REPORT_YEARS:
            raise ValueError(
                f"Unsupported GEP report year: {year}. "
                f"Allowed editions: {sorted(VALID_REPORT_YEARS)}"
            )

        if year not in years:
            years.append(year)

    return sorted(years)


print("JSON/year helpers loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06. Supervisor research-plan validation
# MAGIC
# MAGIC Notebook 03 separated historical observation years from GEP report years.
# MAGIC This agent only consumes `report_years`.

# COMMAND ----------

def validate_research_plan(
    supervisor_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate the research portion of a Supervisor plan."""
    if not isinstance(supervisor_plan, dict):
        raise TypeError(
            "Supervisor plan must be a dictionary."
        )

    route = supervisor_plan.get("route")

    if route not in {
        "rag",
        "temporal_rag",
        "hybrid",
    }:
        raise ValueError(
            "Research Agent only executes rag, temporal_rag, or hybrid plans. "
            f"Received route={route!r}."
        )

    if supervisor_plan.get("needs_research_tool") is not True:
        raise ValueError(
            "Supervisor plan does not authorize research-tool execution."
        )

    original_query = str(
        supervisor_plan.get("original_query", "")
    ).strip()

    resolved_query = str(
        supervisor_plan.get(
            "resolved_query",
            original_query,
        )
    ).strip()

    if not resolved_query:
        raise ValueError(
            "Research execution requires a non-empty resolved query."
        )

    report_years = normalize_years(
        supervisor_plan.get("report_years", [])
    )

    # For temporal comparison, editions must be explicit.
    if route == "temporal_rag" and len(report_years) < 2:
        raise ValueError(
            "temporal_rag requires at least two explicit GEP report years."
        )

    # We intentionally do not hard-filter on region/section because the
    # locked metadata evaluation showed those fields can be incomplete.
    regions = [
        str(x).strip()
        for x in (
            supervisor_plan.get("regions", [])
            or []
        )
        if str(x).strip()
    ]

    return {
        "route": route,
        "original_query": original_query,
        "resolved_query": resolved_query,
        "report_years": report_years,
        "regions": regions,
        "temporal": (
            route == "temporal_rag"
            or len(report_years) > 1
        ),
    }


print("Research-plan validation loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 07. Governed HYBRID retrieval
# MAGIC
# MAGIC The agent cannot choose a different endpoint, index, query type, or arbitrary filter.
# MAGIC Only validated `report_year` filtering is exposed.

# COMMAND ----------

def build_report_year_filter(
    report_years: List[int],
) -> Optional[Dict[str, Any]]:
    """Create only the governed report-year filter."""
    years = normalize_years(report_years)

    if not years:
        return None

    if len(years) == 1:
        return {
            "report_year": years[0]
        }

    return {
        "report_year": years
    }


def _parse_search_response(
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convert Databricks AI Search response rows into dictionaries.

    The response manifest defines the returned column order.
    """
    result = response.get("result", {}) or {}
    manifest = response.get("manifest", {}) or {}

    data_array = result.get("data_array", []) or []
    columns = manifest.get("columns", []) or []

    if not columns:
        raise ValueError(
            "AI Search response is missing manifest columns."
        )

    column_names = []

    for column in columns:
        if isinstance(column, dict):
            name = column.get("name")
        else:
            name = str(column)

        if not name:
            raise ValueError(
                "AI Search returned an unnamed manifest column."
            )

        column_names.append(name)

    parsed = []

    for row in data_array:
        if len(row) != len(column_names):
            raise ValueError(
                "AI Search response row length does not match manifest."
            )

        parsed.append(
            json_safe(
                dict(
                    zip(
                        column_names,
                        row,
                    )
                )
            )
        )

    return parsed


def retrieve_gep_evidence(
    query: str,
    report_years: Optional[List[int]] = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> Dict[str, Any]:
    """Run the locked HYBRID retrieval path."""
    query = str(query).strip()

    if not query:
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    years = normalize_years(
        report_years or []
    )

    top_k = int(top_k)

    if top_k < 1 or top_k > 20:
        raise ValueError(
            "top_k must be between 1 and 20."
        )

    filters = build_report_year_filter(
        years
    )

    kwargs = {
        "query_text": query,
        "columns": RETURN_COLUMNS,
        "num_results": top_k,
        "query_type": "HYBRID",
    }

    if filters:
        kwargs["filters"] = filters

    start = time.perf_counter()

    response = index.similarity_search(
        **kwargs
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    chunks = _parse_search_response(
        response
    )

    # Defense in depth: if years were requested, ensure returned
    # evidence belongs only to those editions.
    if years:
        invalid_year_chunks = [
            chunk
            for chunk in chunks
            if int(chunk["report_year"])
            not in years
        ]

        if invalid_year_chunks:
            raise ValueError(
                "AI Search returned evidence outside the validated "
                "report-year filter."
            )

    return {
        "query": query,
        "report_years": years,
        "filters": filters,
        "query_type": "HYBRID",
        "chunks": chunks,
        "chunk_count": len(chunks),
        "retrieval_latency_ms": round(
            latency_ms,
            2,
        ),
    }


print("Governed HYBRID retrieval loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 08. Context builder
# MAGIC
# MAGIC This follows the locked Notebook 09/10 policy:
# MAGIC - maximum 6 evidence items
# MAGIC - exact duplicate suppression
# MAGIC - near-overlap suppression
# MAGIC - deterministic token budget
# MAGIC - stable `[E#]` evidence IDs

# COMMAND ----------

def _word_set(text: str) -> set:
    return set(
        re.findall(
            r"\b[a-z0-9]+\b",
            (text or "").lower(),
        )
    )


def _jaccard(
    text_a: str,
    text_b: str,
) -> float:
    a = _word_set(text_a)
    b = _word_set(text_b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _truncate_chars(
    text: str,
    max_tokens: int,
) -> str:
    """Deterministic approximation matching the locked context policy."""
    max_chars = int(
        max_tokens * CHARS_PER_TOKEN
    )

    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Avoid ending in the middle of a word where possible.
    last_space = truncated.rfind(" ")

    if last_space > int(max_chars * 0.8):
        truncated = truncated[:last_space]

    return truncated.rstrip() + "…"


def build_research_context(
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build citation-ready evidence and prompt context."""
    accepted = []
    seen_exact = set()

    exact_duplicates_removed = 0
    near_duplicates_removed = 0
    budget_drops = 0

    used_tokens = 0

    for chunk in chunks:
        raw_text = str(
            chunk.get("chunk_text") or ""
        ).strip()

        if not raw_text:
            continue

        exact_key = re.sub(
            r"\s+",
            " ",
            raw_text.lower(),
        ).strip()

        if exact_key in seen_exact:
            exact_duplicates_removed += 1
            continue

        if any(
            _jaccard(
                raw_text,
                item["text"],
            ) >= NEAR_DUPLICATE_JACCARD
            for item in accepted
        ):
            near_duplicates_removed += 1
            continue

        text = _truncate_chars(
            raw_text,
            MAX_TOKENS_PER_EVIDENCE,
        )

        estimated_tokens = max(
            1,
            math.ceil(
                len(text)
                / CHARS_PER_TOKEN
            ),
        )

        if (
            used_tokens + estimated_tokens
            > MAX_CONTEXT_TOKENS
        ):
            budget_drops += 1
            continue

        evidence_id = (
            f"E{len(accepted) + 1}"
        )

        accepted.append({
            "evidence_id": evidence_id,
            "chunk_id": chunk.get(
                "chunk_id"
            ),
            "parent_chunk_id": chunk.get(
                "parent_chunk_id"
            ),
            "document_id": chunk.get(
                "document_id"
            ),
            "report_year": chunk.get(
                "report_year"
            ),
            "edition_status": chunk.get(
                "edition_status"
            ),
            "chapter": chunk.get(
                "chapter"
            ),
            "region": chunk.get(
                "region"
            ),
            "section": chunk.get(
                "section"
            ),
            "subsection": chunk.get(
                "subsection"
            ),
            "content_type": chunk.get(
                "content_type"
            ),
            "page_start": chunk.get(
                "page_start"
            ),
            "page_end": chunk.get(
                "page_end"
            ),
            "text": text,
            "estimated_tokens": estimated_tokens,
        })

        seen_exact.add(exact_key)
        used_tokens += estimated_tokens

        if len(accepted) >= RETRIEVAL_TOP_K:
            break

    if not accepted:
        raise ValueError(
            "No usable evidence remained after context construction."
        )

    blocks = []

    for item in accepted:
        provenance = (
            f"GEP {item['report_year']}; "
            f"document={item['document_id']}; "
            f"pages={item['page_start']}-{item['page_end']}"
        )

        blocks.append(
            f"[{item['evidence_id']}]\n"
            f"Source: {provenance}\n"
            f"{item['text']}"
        )

    context_text = "\n\n".join(
        blocks
    )

    return {
        "evidence": accepted,
        "context_text": context_text,
        "allowed_evidence_ids": [
            item["evidence_id"]
            for item in accepted
        ],
        "estimated_context_tokens": used_tokens,
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicates_removed": near_duplicates_removed,
        "budget_drops": budget_drops,
    }


print("Research context builder loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 09. Grounded research generation

# COMMAND ----------

def generate_grounded_research_answer(
    query: str,
    context_package: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a qualitative research answer using only supplied GEP evidence.
    """
    allowed_ids = context_package[
        "allowed_evidence_ids"
    ]

    prompt = f"""
You are the Research Agent for a World Bank Global Economic Prospects intelligence system.

Answer the user's research question using ONLY the supplied evidence.

Rules:
1. Do not use outside knowledge.
2. Cite every substantive claim using one or more evidence IDs like [E1].
3. Use only these evidence IDs: {allowed_ids}.
4. Do not invent statistics, forecasts, causes, risks, countries, dates, or citations.
5. If the evidence is insufficient for part of the question, say that clearly.
6. When comparing report editions, distinguish what each edition says.
7. Keep the answer factual and concise.
8. Do not produce a final cross-source synthesis with structured macroeconomic data; this is only the qualitative GEP research result.

Question:
{query}

Evidence:
{context_package["context_text"]}
""".strip()

    start = time.perf_counter()

    response = llm_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

    if not answer:
        raise ValueError(
            "Research model returned an empty answer."
        )

    return {
        "answer": answer,
        "generation_latency_ms": round(
            latency_ms,
            2,
        ),
    }


print("Grounded research generation loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Citation validation and provenance mapping

# COMMAND ----------

CITATION_PATTERN = re.compile(
    r"\[(E\d+)\]"
)


def validate_research_citations(
    answer: str,
    context_package: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate every `[E#]` citation against the current evidence package."""
    allowed = set(
        context_package[
            "allowed_evidence_ids"
        ]
    )

    cited = []

    for evidence_id in CITATION_PATTERN.findall(
        answer or ""
    ):
        if evidence_id not in cited:
            cited.append(evidence_id)

    unsupported = [
        evidence_id
        for evidence_id in cited
        if evidence_id not in allowed
    ]

    if not cited:
        raise ValueError(
            "Research answer contains no evidence citations."
        )

    if unsupported:
        raise ValueError(
            "Research answer contains unsupported citations: "
            f"{unsupported}"
        )

    evidence_by_id = {
        item["evidence_id"]: item
        for item in context_package[
            "evidence"
        ]
    }

    provenance = []

    for evidence_id in cited:
        item = evidence_by_id[
            evidence_id
        ]

        provenance.append({
            "evidence_id": evidence_id,
            "chunk_id": item[
                "chunk_id"
            ],
            "parent_chunk_id": item[
                "parent_chunk_id"
            ],
            "document_id": item[
                "document_id"
            ],
            "report_year": item[
                "report_year"
            ],
            "edition_status": item[
                "edition_status"
            ],
            "page_start": item[
                "page_start"
            ],
            "page_end": item[
                "page_end"
            ],
        })

    return {
        "citation_valid": True,
        "cited_evidence_ids": cited,
        "unsupported_citations": unsupported,
        "citation_provenance": provenance,
    }


print("Citation validation loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Research Agent execution
# MAGIC
# MAGIC This is the public agent function used by later orchestration.

# COMMAND ----------

def run_research_agent(
    supervisor_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the qualitative GEP research portion of a Supervisor plan.
    """
    total_start = time.perf_counter()

    plan = validate_research_plan(
        supervisor_plan
    )

    retrieval = retrieve_gep_evidence(
        query=plan["resolved_query"],
        report_years=plan["report_years"],
        top_k=RETRIEVAL_TOP_K,
    )

    if retrieval["chunk_count"] == 0:
        raise ValueError(
            "No GEP evidence was retrieved for the validated research plan."
        )

    context_package = build_research_context(
        retrieval["chunks"]
    )

    generation = generate_grounded_research_answer(
        query=plan["resolved_query"],
        context_package=context_package,
    )

    citation_validation = (
        validate_research_citations(
            generation["answer"],
            context_package,
        )
    )

    total_latency_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    return json_safe({
        "agent": "research_agent",
        "status": "success",
        "route": plan["route"],
        "query": plan["resolved_query"],
        "report_years": plan[
            "report_years"
        ],
        "regions": plan["regions"],
        "answer": generation[
            "answer"
        ],
        "evidence": context_package[
            "evidence"
        ],
        "citation_validation": (
            citation_validation
        ),
        "retrieval": {
            "query_type": retrieval[
                "query_type"
            ],
            "filters": retrieval[
                "filters"
            ],
            "candidate_count": retrieval[
                "chunk_count"
            ],
            "final_evidence_count": len(
                context_package[
                    "evidence"
                ]
            ),
            "rerank_applied": False,
            "rerank_reason": (
                "Locked baseline uses direct HYBRID retrieval; "
                "LLM reranking remains conditional because Notebook 06 "
                "showed modest quality gain with large latency cost."
            ),
        },
        "context": {
            "estimated_tokens": (
                context_package[
                    "estimated_context_tokens"
                ]
            ),
            "exact_duplicates_removed": (
                context_package[
                    "exact_duplicates_removed"
                ]
            ),
            "near_duplicates_removed": (
                context_package[
                    "near_duplicates_removed"
                ]
            ),
            "budget_drops": (
                context_package[
                    "budget_drops"
                ]
            ),
        },
        "latency_ms": {
            "retrieval": retrieval[
                "retrieval_latency_ms"
            ],
            "generation": generation[
                "generation_latency_ms"
            ],
            "total": round(
                total_latency_ms,
                2,
            ),
        },
        "research_contract": {
            "source": (
                "World Bank Global Economic Prospects "
                "January editions 2022-2026"
            ),
            "index": INDEX_NAME,
            "query_type": "HYBRID",
            "historical_structured_data_used": False,
            "outside_knowledge_allowed": False,
            "citations_required": True,
        },
    })


print("run_research_agent() loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Compact Synthesis Agent payload

# COMMAND ----------

def build_research_synthesis_payload(
    research_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return only the grounded research answer and cited provenance needed
    by the later Synthesis Agent.
    """
    if research_result.get(
        "status"
    ) != "success":
        raise ValueError(
            "Cannot build synthesis payload from a failed Research Agent result."
        )

    citation_validation = research_result[
        "citation_validation"
    ]

    if not citation_validation.get(
        "citation_valid"
    ):
        raise ValueError(
            "Research result failed citation validation."
        )

    cited_ids = set(
        citation_validation[
            "cited_evidence_ids"
        ]
    )

    cited_evidence = [
        item
        for item in research_result[
            "evidence"
        ]
        if item["evidence_id"] in cited_ids
    ]

    return {
        "evidence_type": "gep_research",
        "query": research_result[
            "query"
        ],
        "report_years": research_result[
            "report_years"
        ],
        "research_answer": research_result[
            "answer"
        ],
        "cited_evidence": cited_evidence,
        "citation_provenance": (
            citation_validation[
                "citation_provenance"
            ]
        ),
        "citation_valid": True,
    }


print("Research synthesis payload helper loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Test 1 — single-edition RAG route

# COMMAND ----------

single_year_plan = {
    "original_query": (
        "What risks did the World Bank identify for South Asia in 2025?"
    ),
    "resolved_query": (
        "What risks did the World Bank identify for South Asia in 2025?"
    ),
    "route": "rag",
    "countries_or_entities": [],
    "regions": ["South Asia"],
    "indicators": [],
    "observation_start_year": None,
    "observation_end_year": None,
    "report_years": [2025],
    "needs_structured_tool": False,
    "needs_research_tool": True,
}

single_year_result = run_research_agent(
    single_year_plan
)

print(
    json.dumps(
        {
            "status": single_year_result[
                "status"
            ],
            "route": single_year_result[
                "route"
            ],
            "report_years": single_year_result[
                "report_years"
            ],
            "answer": single_year_result[
                "answer"
            ],
            "citation_validation": (
                single_year_result[
                    "citation_validation"
                ]
            ),
            "latency_ms": single_year_result[
                "latency_ms"
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
)

assert single_year_result[
    "status"
] == "success"

assert single_year_result[
    "report_years"
] == [2025]

assert single_year_result[
    "citation_validation"
]["citation_valid"] is True

assert all(
    int(item["report_year"]) == 2025
    for item in single_year_result[
        "evidence"
    ]
)

print("Single-edition Research Agent test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Test 2 — temporal RAG route
# MAGIC
# MAGIC This verifies that the Research Agent can compare multiple GEP editions while
# MAGIC keeping report years separate from structured observation years.

# COMMAND ----------

temporal_plan = {
    "original_query": (
        "How did the World Bank's assessment of global economic risks "
        "change between 2022 and 2026?"
    ),
    "resolved_query": (
        "How did the World Bank's assessment of global economic risks "
        "change between 2022 and 2026?"
    ),
    "route": "temporal_rag",
    "countries_or_entities": [],
    "regions": [],
    "indicators": [],
    "observation_start_year": None,
    "observation_end_year": None,
    "report_years": [2022, 2026],
    "needs_structured_tool": False,
    "needs_research_tool": True,
}

temporal_result = run_research_agent(
    temporal_plan
)

print(
    json.dumps(
        {
            "status": temporal_result[
                "status"
            ],
            "route": temporal_result[
                "route"
            ],
            "report_years": temporal_result[
                "report_years"
            ],
            "answer": temporal_result[
                "answer"
            ],
            "cited_evidence_ids": (
                temporal_result[
                    "citation_validation"
                ][
                    "cited_evidence_ids"
                ]
            ),
            "latency_ms": temporal_result[
                "latency_ms"
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
)

assert temporal_result[
    "report_years"
] == [2022, 2026]

assert temporal_result[
    "citation_validation"
]["citation_valid"] is True

assert all(
    int(item["report_year"])
    in {2022, 2026}
    for item in temporal_result[
        "evidence"
    ]
)

print("Temporal Research Agent test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Test 3 — research portion of a hybrid route

# COMMAND ----------

hybrid_plan = {
    "original_query": (
        "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 "
        "and explain the World Bank outlook for the two regions in 2025."
    ),
    "resolved_query": (
        "Explain the World Bank outlook for South Asia and "
        "Sub-Saharan Africa in the January 2025 Global Economic Prospects."
    ),
    "route": "hybrid",
    "countries_or_entities": [
        "South Asia",
        "Sub-Saharan Africa",
    ],
    "regions": [
        "South Asia",
        "Sub-Saharan Africa",
    ],
    "indicators": [
        "NY.GDP.MKTP.KD.ZG"
    ],
    "observation_start_year": 2022,
    "observation_end_year": 2025,
    "report_years": [2025],
    "needs_structured_tool": True,
    "needs_research_tool": True,
}

hybrid_research_result = run_research_agent(
    hybrid_plan
)

hybrid_payload = (
    build_research_synthesis_payload(
        hybrid_research_result
    )
)

print(
    json.dumps(
        hybrid_payload,
        indent=2,
        ensure_ascii=False,
    )
)

assert hybrid_research_result[
    "route"
] == "hybrid"

assert hybrid_payload[
    "evidence_type"
] == "gep_research"

assert hybrid_payload[
    "citation_valid"
] is True

print("Hybrid research-side execution test passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Guardrail — invalid GEP edition

# COMMAND ----------

invalid_year_guardrail_passed = False

try:
    run_research_agent({
        "original_query": "What did the 2021 GEP say about global risks?",
        "resolved_query": "What did the 2021 GEP say about global risks?",
        "route": "rag",
        "regions": [],
        "report_years": [2021],
        "needs_structured_tool": False,
        "needs_research_tool": True,
    })

except ValueError as exc:
    invalid_year_guardrail_passed = True
    print("Expected invalid-edition rejection:")
    print(str(exc))

assert invalid_year_guardrail_passed

print("Invalid GEP-edition guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Guardrail — reject structured-only plan

# COMMAND ----------

structured_rejection_passed = False

try:
    run_research_agent({
        "original_query": "Show India's GDP growth from 2015 to 2025",
        "resolved_query": "Show India's GDP growth from 2015 to 2025",
        "route": "structured",
        "countries_or_entities": ["India"],
        "indicators": ["NY.GDP.MKTP.KD.ZG"],
        "observation_start_year": 2015,
        "observation_end_year": 2025,
        "report_years": [],
        "needs_structured_tool": True,
        "needs_research_tool": False,
    })

except ValueError as exc:
    structured_rejection_passed = True
    print("Expected structured-route rejection:")
    print(str(exc))

assert structured_rejection_passed

print("Structured-only route rejection passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Guardrail — temporal route requires multiple editions

# COMMAND ----------

temporal_year_guardrail_passed = False

try:
    validate_research_plan({
        "original_query": "How did risks change?",
        "resolved_query": "How did risks change?",
        "route": "temporal_rag",
        "report_years": [2025],
        "needs_research_tool": True,
    })

except ValueError as exc:
    temporal_year_guardrail_passed = True
    print("Expected temporal-plan rejection:")
    print(str(exc))

assert temporal_year_guardrail_passed

print("Temporal-year guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Guardrail — citation validator rejects invented evidence IDs

# COMMAND ----------

citation_guardrail_passed = False

test_context = {
    "allowed_evidence_ids": ["E1"],
    "evidence": [
        {
            "evidence_id": "E1",
            "chunk_id": "test_chunk",
            "parent_chunk_id": "test_parent",
            "document_id": "test_document",
            "report_year": 2025,
            "edition_status": "final",
            "page_start": 1,
            "page_end": 1,
        }
    ],
}

try:
    validate_research_citations(
        "Supported [E1], invented [E99].",
        test_context,
    )

except ValueError as exc:
    citation_guardrail_passed = True
    print("Expected unsupported-citation rejection:")
    print(str(exc))

assert citation_guardrail_passed

print("Citation guardrail passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Agent callable schema

# COMMAND ----------

RESEARCH_AGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_research_agent",
        "description": (
            "Execute the qualitative World Bank Global Economic Prospects "
            "research portion of a validated Supervisor plan using the locked "
            "HYBRID RAG stack and citation validation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "supervisor_plan": {
                    "type": "object",
                    "description": (
                        "Validated Supervisor plan containing route, resolved "
                        "query, GEP report years, and research authorization."
                    ),
                }
            },
            "required": [
                "supervisor_plan"
            ],
            "additionalProperties": False,
        },
    },
}

print(
    json.dumps(
        RESEARCH_AGENT_TOOL_SCHEMA,
        indent=2,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Final validation suite

# COMMAND ----------

validation_results = []


def record_test(
    name: str,
    passed: bool,
    detail: str,
):
    validation_results.append({
        "test": name,
        "passed": bool(passed),
        "detail": detail,
    })


record_test(
    "single_year_rag",
    (
        single_year_result["status"] == "success"
        and single_year_result["report_years"] == [2025]
    ),
    "2025 South Asia risk research executed",
)

record_test(
    "single_year_filter_integrity",
    all(
        int(item["report_year"]) == 2025
        for item in single_year_result["evidence"]
    ),
    "All returned evidence belongs to GEP 2025",
)

record_test(
    "single_year_citations",
    single_year_result[
        "citation_validation"
    ]["citation_valid"],
    "Generated research answer uses valid evidence IDs",
)

record_test(
    "temporal_rag",
    (
        temporal_result["status"] == "success"
        and temporal_result["report_years"] == [2022, 2026]
    ),
    "2022 vs 2026 temporal research executed",
)

record_test(
    "temporal_filter_integrity",
    all(
        int(item["report_year"]) in {2022, 2026}
        for item in temporal_result["evidence"]
    ),
    "Temporal evidence restricted to requested editions",
)

record_test(
    "hybrid_research_side",
    (
        hybrid_research_result["status"] == "success"
        and hybrid_research_result["route"] == "hybrid"
    ),
    "Research side of hybrid route executed",
)

record_test(
    "invalid_edition_guardrail",
    invalid_year_guardrail_passed,
    "Unsupported GEP edition rejected",
)

record_test(
    "structured_route_guardrail",
    structured_rejection_passed,
    "Structured-only plan rejected by Research Agent",
)

record_test(
    "temporal_year_guardrail",
    temporal_year_guardrail_passed,
    "Temporal route requires multiple report editions",
)

record_test(
    "citation_guardrail",
    citation_guardrail_passed,
    "Invented evidence ID rejected",
)

record_test(
    "no_default_reranking",
    (
        single_year_result[
            "retrieval"
        ]["rerank_applied"] is False
    ),
    (
        "Locked HYBRID baseline retained; "
        "high-latency LLM reranking not enabled by default"
    ),
)

record_test(
    "synthesis_payload",
    (
        hybrid_payload[
            "evidence_type"
        ] == "gep_research"
        and hybrid_payload[
            "citation_valid"
        ] is True
    ),
    "Citation-ready research payload built for Synthesis Agent",
)

validation_df = spark.createDataFrame(
    validation_results
)

display(validation_df)

failed_tests = [
    row["test"]
    for row in validation_results
    if not row["passed"]
]

assert not failed_tests, (
    f"Research Agent validation failures: {failed_tests}"
)

print("All Research Agent validation tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Final status

# COMMAND ----------

print("")
print("05_research_agent COMPLETE")
print("")
print("Validated path:")
print(
    "Supervisor research plan -> validation -> governed HYBRID AI Search -> "
    "report-year filtering -> context builder -> grounded Llama generation -> "
    "citation validation -> research synthesis payload"
)
print("")
print("Important:")
print(" - Uses the locked structure-aware Qwen3 AI Search index")
print(" - Uses HYBRID retrieval")
print(" - Only GEP January editions 2022-2026 are allowed")
print(" - 2026 remains an advance edition in source metadata")
print(" - Region metadata is not used as a hard filter")
print(" - No arbitrary index or filter selection is exposed")
print(" - No structured macroeconomic lookup occurs here")
print(" - No default LLM reranking")
print(" - Every generated research answer must cite valid [E#] evidence")
print(" - Final cross-source synthesis is deferred to Notebook 06")
print("")
print("Next notebook after this passes:")
print("05_tools_agents / 06_synthesis_agent")