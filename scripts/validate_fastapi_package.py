from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = [
        ROOT / "api" / "main.py",
        ROOT / "api" / "schemas.py",
        ROOT / "api" / "settings.py",
        ROOT / "api" / "pdf_processing.py",
        ROOT / "api" / "errors.py",
        ROOT / "src" / "job_optimizer_engine.py",
        ROOT / "scripts" / "validate_fastapi_backend.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    main_source = paths[0].read_text(encoding="utf-8")
    settings_source = paths[2].read_text(encoding="utf-8")
    required = {
        "lifespan=lifespan": main_source,
        "asyncio.Semaphore(1)": main_source,
        "read_upload_limited": main_source,
        "response_model=ResumeAnalysisResponse": main_source,
        "X-Request-ID": main_source,
        "Cache-Control": main_source,
        'if "*" in origins': settings_source,
        'environment == "production"': settings_source,
    }
    missing = [marker for marker, source in required.items() if marker not in source]
    if missing:
        raise ValueError(f"Required API safeguards are missing: {missing}")

    forbidden = [
        "allow_origins=[\"*\"]",
        "allow_origins=['*']",
        "torch.save(",
        ".backward(",
        "optimizer.step(",
        "requests.get(",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    present = [marker for marker in forbidden if marker in combined]
    if present:
        raise ValueError(f"Forbidden API or training patterns detected: {present}")

    print("FastAPI package validation passed.")
    print("Syntax: passed")
    print("Upload and concurrency limits: present")
    print("CORS wildcard: absent")
    print("Training and download logic: absent")


if __name__ == "__main__":
    main()
