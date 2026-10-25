from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import SELECTED_MODEL_NAME
from backend.services.data_service import DataService
from backend.services.model_service import ModelService
from backend.services.pipeline_results_service import PipelineResultsService

app = FastAPI(title="Business Entity Resolution API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):517\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
data_service = DataService()
pipeline_results_service = PipelineResultsService(data_service, model_service)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    train_counts = data_service.get_train_counts()
    test_counts = data_service.get_test_counts()
    dashboard_summary = data_service.get_dashboard_summary()
    submission_summary = pipeline_results_service.get_submission_summary()

    return {
        "source1_train_count": train_counts["source1_train_count"],
        "source2_train_count": train_counts["source2_train_count"],
        "source3_train_count": train_counts["source3_train_count"],
        "source1_test_count": test_counts.get("source1_test_count"),
        "source2_test_count": test_counts.get("source2_test_count"),
        "source3_test_count": test_counts.get("source3_test_count"),
        "training_candidate_count": dashboard_summary.get("training_candidate_count", 0),
        "test_candidate_count": submission_summary["total_candidate_pairs"],
        "test_predicted_matches": submission_summary["total_matches"],
        "blocking_recall": dashboard_summary.get("blocking_recall", 0.0),
        "search_space_reduction": dashboard_summary.get("search_space_reduction", 0.0),
        "positive_matches": dashboard_summary.get("positive_matches", 0),
        "selected_model": model_service.get_selected_model_name(),
        "selected_threshold": model_service.get_selected_threshold(),
    }


@app.get("/api/models")
def model_metrics() -> list[dict[str, Any]]:
    return model_service.get_model_metrics()


@app.get("/api/entities/search")
def search_entities(q: str = Query(default="", min_length=1)) -> list[dict[str, str]]:
    return data_service.search_entities(q, limit=25)


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str) -> dict[str, str]:
    try:
        return data_service.get_entity(entity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/entities/{entity_id}/candidates")
def get_candidates(entity_id: str) -> list[dict[str, Any]]:
    try:
        return pipeline_results_service.get_candidates_for_entity(entity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/entities/{entity_id}/matches")
def get_matches(entity_id: str) -> list[dict[str, Any]]:
    try:
        return pipeline_results_service.get_entity_matches(entity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/submission")
def submission_summary() -> dict[str, Any]:
    return pipeline_results_service.get_submission_summary()
