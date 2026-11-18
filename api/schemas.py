from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: Literal["live", "ready"]
    service: str
    api_version: str


class ReadyResponse(HealthResponse):
    model_version: str
    ranking_policy: str
    rankable_job_count: int


class ModelInfoResponse(StrictModel):
    model_name: str
    model_version: str
    ranking_policy: str
    gnn_weight: float
    sbert_weight: float
    rankable_job_count: int
    score_meaning: str
    limitations: list[str]


class CandidateSummary(StrictModel):
    detected_skills: list[str]
    detected_skill_count: int
    resume_character_count: int


class SkillEvidence(StrictModel):
    confidence: Literal["high", "moderate", "limited", "unavailable"]
    structured_evidence_available: bool
    required_skill_count: int
    preferred_skill_count: int
    mentioned_skill_count: int
    unique_structured_skill_count: int
    unique_matched_skill_count: int
    overlap_ratio: float | None
    matched_required: list[str]
    matched_preferred: list[str]
    matched_mentioned: list[str]
    missing_required: list[str]
    missing_preferred: list[str]


class JobMatch(StrictModel):
    rank: int
    job_id: str
    title: str
    company: str
    location: str
    url: str | None
    description: str
    predicted_compatibility_score: float = Field(ge=0, le=100)
    predicted_skills_score: float = Field(ge=0, le=100)
    gnn_composite_score: float
    calibrated_semantic_score: float
    semantic_cosine_similarity: float = Field(ge=-1, le=1)
    evidence: SkillEvidence


class ResumeAnalysisResponse(StrictModel):
    request_id: str
    api_version: str
    model_version: str
    ranking_policy: str
    processing_time_ms: int
    candidate: CandidateSummary
    matches: list[JobMatch]
    disclaimer: str


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorDetail
