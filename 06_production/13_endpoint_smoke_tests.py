# Databricks notebook source
# MAGIC %md
# MAGIC # 13 - Production endpoint smoke tests
# MAGIC Test one request for each route before connecting the UI.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json
import time

ENDPOINT_NAME = "worldbank-gep-intelligence-agent"
w = WorkspaceClient()

tests = [
    ("structured", "Show India's GDP growth from 2015 to 2025."),
    ("rag", "What risks did the January 2025 Global Economic Prospects report highlight?"),
    ("temporal_rag", "How did downside risks change between the January 2022 and January 2025 GEP reports?"),
    ("hybrid", "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 and explain the January 2025 outlook."),
]

for expected_route, question in tests:
    started = time.perf_counter()

    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[
            {
                "question": question,
                "conversation_context": None,
            }
        ],
    )

    payload = response.as_dict()
    predictions = payload.get("predictions") or []
    assert predictions, f"No prediction returned for: {question}"

    result = predictions[0]

    print("=" * 90)
    print("Expected route:", expected_route)
    print("Actual route:", result.get("route"))
    print("Endpoint latency ms:", round((time.perf_counter() - started) * 1000, 2))
    print("Answer:")
    print(result.get("answer"))

    assert result.get("route") == expected_route

print("\nAll endpoint route smoke tests PASSED.")
