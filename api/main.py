from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.errors import ApiError
from api.pdf_processing import PdfProcessingError, extract_pdf_text
from api.schemas import (
    CandidateSummary,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    JobMatch,
    ModelInfoResponse,
    ReadyResponse,
    ResumeAnalysisResponse,
    SkillEvidence,
)
from api.settings import get_settings
from src.job_optimizer_engine import JobOptimizerEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_VERSION = "v1"
SERVICE_NAME = "Job Optimizer API"
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = await asyncio.to_thread(JobOptimizerEngine, PROJECT_ROOT)
    if engine.ranking_policy is None:
        raise RuntimeError("The validated production ranking policy is unavailable")
    app.state.engine = engine
    app.state.inference_semaphore = asyncio.Semaphore(1)
    yield
    app.state.engine = None


app = FastAPI(
    title=SERVICE_NAME,
    version="1.0.0",
    docs_url=None if settings.environment == "production" else "/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
    expose_headers=["X-Request-ID"],
    max_age=600,
)


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def error_response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id(request),
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return error_response(request, exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return error_response(
        request,
        422,
        "invalid_request",
        "The request parameters or uploaded form data are invalid.",
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    return error_response(
        request,
        500,
        "internal_error",
        "The request could not be completed.",
    )


async def read_upload_limited(upload: UploadFile) -> bytes:
    content = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > settings.maximum_upload_bytes:
            raise ApiError(
                413,
                "file_too_large",
                f"The PDF exceeds the {settings.maximum_upload_bytes // (1024 * 1024)} MB limit.",
            )
    if not content:
        raise ApiError(400, "empty_file", "The uploaded PDF is empty.")
    return bytes(content)


def evidence_confidence(match: dict[str, Any]) -> str:
    count = int(match["unique_structured_skill_count"])
    if count >= 10:
        return "high"
    if count >= 5:
        return "moderate"
    if count > 0:
        return "limited"
    return "unavailable"


def public_match(match: dict[str, Any]) -> JobMatch:
    semantic_score = match.get("sbert_calibrated_composite_score")
    if semantic_score is None:
        raise RuntimeError("Production hybrid inference returned no semantic score")
    evidence = SkillEvidence(
        confidence=evidence_confidence(match),
        structured_evidence_available=bool(
            match["structured_skill_evidence_available"]
        ),
        required_skill_count=int(match["required_skill_count"]),
        preferred_skill_count=int(match["preferred_skill_count"]),
        mentioned_skill_count=int(match["mentioned_skill_count"]),
        unique_structured_skill_count=int(match["unique_structured_skill_count"]),
        unique_matched_skill_count=int(match["unique_matched_skill_count"]),
        overlap_ratio=match["structured_skill_overlap_ratio"],
        matched_required=match["matched_required_skills"],
        matched_preferred=match["matched_preferred_skills"],
        matched_mentioned=match["matched_mentioned_skills"],
        missing_required=match["missing_required_skills"],
        missing_preferred=match["missing_preferred_skills"],
    )
    return JobMatch(
        rank=int(match["rank"]),
        job_id=str(match["job_id"]),
        title=str(match["title"]),
        company=str(match["company"]),
        location=str(match["location"]),
        url=match["url"],
        description=str(match["description"]),
        predicted_compatibility_score=float(match["display_composite_score"]),
        predicted_skills_score=float(match["display_skills_score"]),
        gnn_composite_score=float(match["raw_composite_score"]),
        calibrated_semantic_score=float(semantic_score),
        semantic_cosine_similarity=float(match["job_feature_cosine_similarity"]),
        evidence=evidence,
    )


@app.get("/api/v1/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    return HealthResponse(
        status="live",
        service=SERVICE_NAME,
        api_version=API_VERSION,
    )


@app.get("/api/v1/health/ready", response_model=ReadyResponse)
async def health_ready(request: Request) -> ReadyResponse:
    engine: JobOptimizerEngine | None = getattr(request.app.state, "engine", None)
    if engine is None or engine.ranking_policy is None:
        raise ApiError(503, "not_ready", "The matching engine is not ready.")
    return ReadyResponse(
        status="ready",
        service=SERVICE_NAME,
        api_version=API_VERSION,
        model_version=str(engine.ranking_policy["version"]),
        ranking_policy=str(engine.ranking_policy["policy_name"]),
        rankable_job_count=len(engine.main_job_ids),
    )


@app.get("/api/v1/model", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    engine: JobOptimizerEngine = request.app.state.engine
    policy = engine.ranking_policy
    if policy is None:
        raise ApiError(503, "not_ready", "The production ranking policy is unavailable.")
    return ModelInfoResponse(
        model_name=str(engine.config.get("model_name", "Job Optimizer")),
        model_version=str(policy["version"]),
        ranking_policy=str(policy["policy_name"]),
        gnn_weight=float(policy["weights"]["gnn"]),
        sbert_weight=float(policy["weights"]["sbert"]),
        rankable_job_count=len(engine.main_job_ids),
        score_meaning=(
            "Predicted profile-to-job compatibility out of 100; "
            "not a hiring probability."
        ),
        limitations=[
            "Structured skill evidence is unavailable for some job records.",
            "Scores are compatibility estimates and may be closely grouped.",
            "Results do not represent recruiter or employer decisions.",
        ],
    )


@app.post(
    "/api/v1/analyze-resume",
    response_model=ResumeAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_resume(
    request: Request,
    file: Annotated[UploadFile, File(description="Text-based PDF resume")],
    top_k: Annotated[int, Query(ge=1, le=25)] = 10,
) -> ResumeAnalysisResponse:
    filename = Path(file.filename or "").name
    if Path(filename).suffix.casefold() != ".pdf":
        raise ApiError(415, "unsupported_file_type", "Only PDF resumes are supported.")
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise ApiError(
            415,
            "unsupported_media_type",
            "The upload must use the application/pdf media type.",
        )

    try:
        content = await read_upload_limited(file)
    finally:
        await file.close()
    try:
        resume_text = await asyncio.to_thread(
            extract_pdf_text,
            content,
            settings.maximum_pdf_pages,
            settings.maximum_resume_characters,
        )
    except PdfProcessingError as exc:
        raise ApiError(400, "invalid_pdf", str(exc)) from exc

    acquired = False
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            request.app.state.inference_semaphore.acquire(),
            timeout=settings.inference_queue_timeout_seconds,
        )
        acquired = True
        engine: JobOptimizerEngine = request.app.state.engine
        result = await asyncio.to_thread(engine.rank_resume, resume_text, top_k)
    except TimeoutError as exc:
        raise ApiError(
            503,
            "inference_busy",
            "The matching engine is busy. Please try again shortly.",
        ) from exc
    finally:
        if acquired:
            request.app.state.inference_semaphore.release()

    processing_time_ms = max(1, round((time.perf_counter() - started) * 1000))
    return ResumeAnalysisResponse(
        request_id=request_id(request),
        api_version=API_VERSION,
        model_version=str(request.app.state.engine.ranking_policy["version"]),
        ranking_policy=str(result["ranking_policy"]),
        processing_time_ms=processing_time_ms,
        candidate=CandidateSummary(
            detected_skills=result["candidate"]["detected_skills"],
            detected_skill_count=int(result["candidate"]["detected_skill_count"]),
            resume_character_count=int(result["candidate"]["resume_character_count"]),
        ),
        matches=[public_match(match) for match in result["matches"]],
        disclaimer=str(result["disclaimer"]),
    )
