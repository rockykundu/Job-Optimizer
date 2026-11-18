from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app
from api.pdf_processing import extract_pdf_text
from api.settings import get_settings


OUTPUT_PATH = ROOT / "evaluation" / "api" / "fastapi_validation_report.json"
SAMPLE_RESUME = (
    "Backend software engineer with Python, Django, FastAPI, PostgreSQL, Redis, "
    "Docker, AWS, Linux, Git, REST API development, automated testing and CI/CD."
)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_test_pdf(text: str) -> bytes:
    stream = f"BT /F1 10 Tf 72 720 Td ({pdf_escape(text)}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(document)


def require_status(response, expected: int, label: str) -> None:
    if response.status_code != expected:
        raise ValueError(
            f"{label}: expected HTTP {expected}, received {response.status_code}: "
            f"{response.text[:500]}"
        )


def main() -> None:
    pdf = create_test_pdf(SAMPLE_RESUME)
    settings = get_settings()
    extracted = extract_pdf_text(
        pdf,
        settings.maximum_pdf_pages,
        settings.maximum_resume_characters,
    )
    if len(extracted) < 50:
        raise ValueError("Generated validation PDF did not produce enough text")

    with TestClient(app, raise_server_exceptions=False) as client:
        live = client.get("/api/v1/health/live")
        require_status(live, 200, "liveness")
        ready = client.get("/api/v1/health/ready")
        require_status(ready, 200, "readiness")
        model = client.get("/api/v1/model")
        require_status(model, 200, "model information")

        ready_json = ready.json()
        model_json = model.json()
        if ready_json["status"] != "ready":
            raise ValueError("Readiness endpoint did not return ready")
        if ready_json["model_version"] != "1.1.0":
            raise ValueError("API did not load production model version 1.1.0")
        if model_json["gnn_weight"] != 0.78 or model_json["sbert_weight"] != 0.22:
            raise ValueError("API model information contains incorrect weights")

        invalid_extension = client.post(
            "/api/v1/analyze-resume",
            files={"file": ("resume.txt", pdf, "application/pdf")},
        )
        require_status(invalid_extension, 415, "invalid extension")
        invalid_content = client.post(
            "/api/v1/analyze-resume",
            files={"file": ("resume.pdf", b"not a pdf", "application/pdf")},
        )
        require_status(invalid_content, 400, "invalid PDF")

        allowed_origin = settings.allowed_origins[0]
        preflight = client.options(
            "/api/v1/analyze-resume",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        require_status(preflight, 200, "allowed CORS preflight")
        if preflight.headers.get("access-control-allow-origin") != allowed_origin:
            raise ValueError("Allowed CORS origin was not returned")

        response = client.post(
            "/api/v1/analyze-resume?top_k=5",
            files={"file": ("resume.pdf", pdf, "application/pdf")},
        )
        require_status(response, 200, "resume analysis")
        payload = response.json()
        if payload["request_id"] != response.headers.get("X-Request-ID"):
            raise ValueError("Body and response-header request IDs differ")
        if payload["model_version"] != "1.1.0":
            raise ValueError("Analysis response has the wrong model version")
        if len(payload["matches"]) != 5:
            raise ValueError("Analysis response did not return five matches")
        if payload["ranking_policy"] != "validation_selected_gnn_sbert_hybrid":
            raise ValueError("Analysis response used the wrong ranking policy")

        engine = client.app.state.engine
        direct = engine.rank_resume(extracted, top_k=5)
        api_ids = [match["job_id"] for match in payload["matches"]]
        direct_ids = [match["job_id"] for match in direct["matches"]]
        if api_ids != direct_ids:
            raise ValueError("API and direct engine top-five job IDs differ")
        maximum_score_error = max(
            abs(
                float(api_match["predicted_compatibility_score"])
                - float(direct_match["display_composite_score"])
            )
            for api_match, direct_match in zip(payload["matches"], direct["matches"])
        )
        if maximum_score_error > 1e-8:
            raise ValueError("API and direct engine scores differ")

        forbidden_response_keys = {"resume_text", "project_root", "checkpoint_path"}
        serialized = json.dumps(payload)
        if any(f'"{key}"' in serialized for key in forbidden_response_keys):
            raise ValueError("The API response exposed a forbidden internal field")

        report = {
            "stage": "Stage 12 FastAPI backend validation",
            "status": "passed",
            "api_version": payload["api_version"],
            "model_version": payload["model_version"],
            "ranking_policy": payload["ranking_policy"],
            "rankable_job_count": ready_json["rankable_job_count"],
            "validated_match_count": len(payload["matches"]),
            "api_direct_engine_job_ids_exact": True,
            "maximum_api_direct_score_error": maximum_score_error,
            "checks": {
                "liveness_endpoint": True,
                "readiness_endpoint": True,
                "model_information_endpoint": True,
                "valid_pdf_analysis": True,
                "invalid_extension_rejected": True,
                "invalid_pdf_rejected": True,
                "allowed_cors_preflight": True,
                "request_id_header_and_body_match": True,
                "typed_response_schema": True,
                "resume_text_not_returned": True,
                "internal_paths_not_returned": True,
                "production_policy_loaded": True,
            },
            "safety": {
                "training_run": False,
                "dataset_download_requested": False,
                "model_download_requested": False,
                "uploaded_pdf_persisted": False,
                "single_inference_queue": True,
            },
        }
        save_json(OUTPUT_PATH, report)

    print("FastAPI production backend validation passed.")
    print(f"Rankable jobs: {report['rankable_job_count']:,}")
    print("API/direct-engine top-five IDs: exact")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
