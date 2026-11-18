from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _positive_integer(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    environment: str
    allowed_origins: tuple[str, ...]
    maximum_upload_bytes: int
    maximum_pdf_pages: int
    maximum_resume_characters: int
    inference_queue_timeout_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    environment = os.getenv("JOB_OPTIMIZER_ENV", "development").strip().lower()
    origins = tuple(
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "JOB_OPTIMIZER_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )
    if not origins:
        raise ValueError("JOB_OPTIMIZER_ALLOWED_ORIGINS must not be empty")
    if "*" in origins:
        raise ValueError("Wildcard CORS origins are not permitted")
    if environment == "production" and any(
        origin.startswith("http://") for origin in origins
    ):
        raise ValueError("Production CORS origins must use HTTPS")

    return Settings(
        environment=environment,
        allowed_origins=origins,
        maximum_upload_bytes=_positive_integer(
            "JOB_OPTIMIZER_MAX_UPLOAD_BYTES", 10 * 1024 * 1024
        ),
        maximum_pdf_pages=_positive_integer("JOB_OPTIMIZER_MAX_PDF_PAGES", 25),
        maximum_resume_characters=_positive_integer(
            "JOB_OPTIMIZER_MAX_RESUME_CHARACTERS", 100_000
        ),
        inference_queue_timeout_seconds=_positive_integer(
            "JOB_OPTIMIZER_INFERENCE_QUEUE_TIMEOUT_SECONDS", 120
        ),
    )
