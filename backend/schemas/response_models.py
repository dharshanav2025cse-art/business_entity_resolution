from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class DashboardResponse(BaseModel):
    source1_train_count: int
    source2_train_count: int
    source3_train_count: int
    source1_test_count: int | None
    source2_test_count: int | None
    source3_test_count: int | None
    training_candidate_count: int
    test_candidate_count: int | None = None
    blocking_recall: float
    search_space_reduction: float
    positive_matches: int
    selected_model: str
    selected_threshold: float


class ModelMetric(BaseModel):
    name: str
    precision: float
    recall: float
    f05: float
    pr_auc: float
    false_positives: int
    false_negatives: int


class SearchResult(BaseModel):
    entity_id: str
    business_name: str
    business_address: str
    country: str


class CandidateResult(BaseModel):
    candidate_entity_id: str
    source: str
    business_name: str
    business_address: str
    country: str


class MatchResult(BaseModel):
    candidate_entity_id: str
    source: str
    business_name: str
    business_address: str
    country: str
    probability: float
    is_match: bool
    features: dict[str, Any]
