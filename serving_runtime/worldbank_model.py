# ============================================================
# World Bank GEP Intelligence Agent
# MLflow production model
# ============================================================
#
# This file is the entry point packaged by MLflow.
#
# Request
#   ↓
# MLflow PyFunc
#   ↓
# execute_agent()
#   ↓
# Supervisor
#   ↓
# Guardrails
#   ↓
# Data Agent / Research Agent
#   ↓
# Synthesis Agent
#
# IMPORTANT:
# Because this file is logged using MLflow "Models from Code",
# mlflow.models.set_model() MUST be called at the bottom.
# ============================================================


import json
from typing import Any

import mlflow
import pandas as pd

from mlflow.pyfunc import PythonModel
from mlflow.models import set_model


# ============================================================
# JSON helper
# ============================================================

def _to_json(value: Any) -> str:
    """
    Serialize agent objects safely for the stable serving schema.
    """

    if value is None:
        return "null"

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# MLflow model
# ============================================================

class WorldBankGEPAgentModel(PythonModel):
    """
    MLflow wrapper around the tested World Bank GEP
    Intelligence Agent orchestration runtime.
    """

    def load_context(self, context):
        """
        Load the orchestration layer when MLflow loads the model.

        The runtime modules are packaged with the model through
        code_paths in the registration notebook.
        """

        from serving_runtime.orchestrator import execute_agent

        self.execute_agent = execute_agent


    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params=None,
    ) -> pd.DataFrame:
        """
        Execute one or more agent requests.

        Expected input columns:

            question
            conversation_context

        Stable output columns:

            status
            route
            answer
            total_latency_ms
            plan_json
            citation_validation_json
            structured_json
            research_json
        """

        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(
                "model_input must be a pandas DataFrame."
            )

        if "question" not in model_input.columns:
            raise ValueError(
                "Input must contain a 'question' column."
            )


        outputs = []


        # ----------------------------------------------------
        # Process every request row
        # ----------------------------------------------------

        for _, row in model_input.iterrows():

            question = row.get("question")

            if question is None:
                raise ValueError(
                    "question cannot be null."
                )

            question = str(question).strip()

            if not question:
                raise ValueError(
                    "question cannot be empty."
                )


            # ------------------------------------------------
            # Conversation context
            # ------------------------------------------------

            conversation_context = row.get(
                "conversation_context"
            )

            # Pandas can represent missing values as NaN.
            if pd.isna(conversation_context):
                conversation_context = None

            elif conversation_context is not None:
                conversation_context = str(
                    conversation_context
                )


            # ------------------------------------------------
            # Execute complete agent orchestration
            # ------------------------------------------------

            result = self.execute_agent(
                question=question,
                conversation_context=conversation_context,
            )


            if not isinstance(result, dict):
                raise TypeError(
                    "execute_agent() must return a dictionary."
                )


            # ------------------------------------------------
            # Stable serving response
            # ------------------------------------------------

            outputs.append(
                {
                    "status": str(
                        result.get(
                            "status",
                            "unknown",
                        )
                    ),

                    "route": str(
                        result.get(
                            "route",
                            "unknown",
                        )
                    ),

                    "answer": str(
                        result.get(
                            "answer",
                            "",
                        )
                    ),

                    "total_latency_ms": float(
                        result.get(
                            "total_latency_ms",
                            0.0,
                        )
                        or 0.0
                    ),

                    "plan_json": _to_json(
                        result.get(
                            "plan"
                        )
                    ),

                    "citation_validation_json": _to_json(
                        result.get(
                            "citation_validation"
                        )
                    ),

                    "structured_json": _to_json(
                        result.get(
                            "data_agent_result"
                        )
                    ),

                    "research_json": _to_json(
                        result.get(
                            "research_agent_result"
                        )
                    ),
                }
            )


        return pd.DataFrame(outputs)


# ============================================================
# MLflow Models-from-Code registration
# ============================================================
#
# This is REQUIRED when log_model() receives the path to this
# Python file:
#
#     python_model=".../worldbank_model.py"
#
# MLflow executes this file and retrieves the model instance
# registered below.
# ============================================================

set_model(
    WorldBankGEPAgentModel()
)