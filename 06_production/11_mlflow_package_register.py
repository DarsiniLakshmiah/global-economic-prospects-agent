# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 11 - Package and register the agent with MLflow
# MAGIC Uses MLflow PyFunc because the application is custom orchestration code, not a newly trained LLM.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.12.0" "databricks-sdk>=0.102.0" databricks-openai databricks-ai-search pandas pydantic

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# 11 - MLflow Package + Unity Catalog Registration
# ============================================================

import os

import mlflow
import pandas as pd

from mlflow.models import infer_signature


# ============================================================
# 1. MLflow / Unity Catalog configuration
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)

CATALOG = "worldbank_ai"

MODEL_NAME = (
    f"{CATALOG}.ai.gep_intelligence_agent"
)


# ============================================================
# 2. Runtime paths
# ============================================================
#
# Our workspace structure:
#
# global-economic-prospects-agent/
#
# ├── 06_production/
# │   └── 11_mlflow_package_register
# │
# └── serving_runtime/
#     ├── supervisor_runtime.py
#     ├── data_agent_sql_runtime.py
#     ├── research_agent_runtime.py
#     ├── synthesis_agent_runtime.py
#     ├── guardrails_runtime.py
#     ├── orchestrator.py
#     └── worldbank_model.py
#
# ============================================================

RUNTIME_DIR = (
    "/Workspace/Users/"
    "darsinilakshmiah@gmail.com/"
    "global-economic-prospects-agent/"
    "serving_runtime"
)

MODEL_FILE = os.path.join(
    RUNTIME_DIR,
    "worldbank_model.py",
)


# ============================================================
# 3. Validate files before MLflow packaging
# ============================================================

assert os.path.isdir(
    RUNTIME_DIR
), (
    f"Runtime directory does not exist: "
    f"{RUNTIME_DIR}"
)

assert os.path.isfile(
    MODEL_FILE
), (
    f"Model file does not exist: "
    f"{MODEL_FILE}"
)


required_runtime_files = [
    "supervisor_runtime.py",
    "data_agent_sql_runtime.py",
    "research_agent_runtime.py",
    "synthesis_agent_runtime.py",
    "guardrails_runtime.py",
    "orchestrator.py",
    "worldbank_model.py",
]


for filename in required_runtime_files:

    filepath = os.path.join(
        RUNTIME_DIR,
        filename,
    )

    assert os.path.isfile(
        filepath
    ), (
        f"Missing runtime file: "
        f"{filepath}"
    )


print(
    "Runtime package validation PASSED."
)

print(
    "Runtime directory:",
    RUNTIME_DIR,
)

print(
    "Model file:",
    MODEL_FILE,
)


# ============================================================
# 4. Input example
# ============================================================

input_example = pd.DataFrame(
    [
        {
            "question": (
                "Show India's GDP growth "
                "from 2015 to 2025."
            ),

            "conversation_context": None,
        }
    ]
)


# ============================================================
# 5. Output schema example
# ============================================================
#
# This defines the serving contract without executing the live
# agent during signature inference.
# ============================================================

output_example = pd.DataFrame(
    [
        {
            "status": "success",

            "route": "structured",

            "answer": (
                "example schema only"
            ),

            "total_latency_ms": 0.0,

            "plan_json": "{}",

            "citation_validation_json": "null",

            "structured_json": "{}",

            "research_json": "null",
        }
    ]
)


# ============================================================
# 6. Explicit MLflow signature
# ============================================================

signature = infer_signature(
    input_example,
    output_example,
)


print(
    "MLflow model signature created."
)


# ============================================================
# 7. Production dependencies
# ============================================================

pip_requirements = [

    "mlflow>=3.12.0",

    "pandas",

    "pydantic",

    "databricks-sdk>=0.102.0",

    "databricks-openai",

    "databricks-ai-search",
]


# ============================================================
# 8. Log MLflow Model from Code
# ============================================================
#
# worldbank_model.py contains:
#
#     mlflow.models.set_model(...)
#
# which is required for the Models-from-Code workflow.
#
# code_paths packages the rest of serving_runtime with the
# model.
# ============================================================

with mlflow.start_run(
    run_name=(
        "gep-intelligence-agent-production"
    )
):

    model_info = (
        mlflow.pyfunc.log_model(

            name="agent",

            # Models-from-Code entry point.
            python_model=MODEL_FILE,

            # Package the complete runtime.
            code_paths=[
                RUNTIME_DIR
            ],

            input_example=(
                input_example
            ),

            signature=signature,

            pip_requirements=(
                pip_requirements
            ),
        )
    )


print(
    "MLflow model logged successfully."
)

print(
    "Logged model URI:",
    model_info.model_uri,
)


# ============================================================
# 9. Register in Unity Catalog
# ============================================================

registered = mlflow.register_model(
    model_uri=(
        model_info.model_uri
    ),

    name=MODEL_NAME,
)


# ============================================================
# 10. Final validation output
# ============================================================

print(
    "\n=========================================="
)

print(
    "MODEL REGISTRATION PASSED"
)

print(
    "=========================================="
)

print(
    "Registered model:",
    MODEL_NAME,
)

print(
    "Version:",
    registered.version,
)

print(
    "Model URI:",
    model_info.model_uri,
)