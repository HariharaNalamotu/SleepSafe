"""Databricks App entry point. /classify is strict JSON; /ask returns grounded text."""
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sleepsafe.agent import answer_question
from sleepsafe.classifier import classify_sleep_session
from sleepsafe.schema import validate_output
from sleepsafe.repository import DeltaRepository
from sleepsafe.tools import set_repository

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
app = FastAPI(title="SleepSafe", version="0.1.0")
if os.getenv("SLEEPSAFE_REPOSITORY") == "delta":
    set_repository(DeltaRepository())

try:
    import mlflow
    mlflow.autolog(disable=False)
    classify_sleep_session = mlflow.trace(classify_sleep_session)
    answer_question = mlflow.trace(answer_question)
except Exception:
    logging.getLogger(__name__).warning("MLflow tracing unavailable", exc_info=True)


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classify/{session_id}")
def classify(session_id: str):
    try:
        return validate_output(classify_sleep_session(session_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return {"session_id": request.session_id,
                "answer": answer_question(request.session_id, request.question)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
