# Stage 12 — Production FastAPI Backend

This package adds the versioned HTTP backend that the React/TypeScript website
will consume.

## Endpoints

- `GET /api/v1/health/live`
- `GET /api/v1/health/ready`
- `GET /api/v1/model`
- `POST /api/v1/analyze-resume?top_k=10`
- Development OpenAPI UI: `GET /api/docs`
- OpenAPI schema: `GET /api/openapi.json`

## Security and resource controls

- PDF extension, media type and file signature validation.
- 10 MB default upload limit.
- 25-page default limit.
- 100,000-character extracted-text limit.
- Uploaded PDFs and resume text remain in memory and are not persisted.
- Scanned image-only and password-protected PDFs are rejected.
- Wildcard CORS is prohibited.
- Production origins must use HTTPS.
- One inference runs at a time to protect CPU and memory.
- Public errors do not expose internal paths or exception details.
- Every response receives a server-generated request ID.

Authentication, public rate limiting and TLS termination belong to the later
deployment layer. Until those controls are added, bind the development server
to `127.0.0.1`, not `0.0.0.0`.

## Install

From the project root:

```powershell
Copy-Item .\src\job_optimizer_engine.py .\src\job_optimizer_engine.before-api.py.bak -Force
```

```powershell
Expand-Archive -Path .\JobOptimizer_Stage12_FastAPIBackend.zip -DestinationPath . -Force
```

```powershell
python -m pip install -r .\requirements-api.txt
```

The pip command installs Python API libraries only. It does not download the
dataset or ML model.

## Validate

```powershell
python .\scripts\validate_fastapi_package.py
```

```powershell
python .\scripts\validate_fastapi_backend.py
```

The runtime validation loads the production engine, exercises health, model,
CORS, invalid-upload and real PDF-analysis paths, then compares the HTTP result
with direct engine inference.

Report:

`evaluation\api\fastapi_validation_report.json`

## Local development server

Start only after validation passes:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open:

`http://127.0.0.1:8000/api/docs`

Use one Uvicorn worker because each worker would load another complete copy of
the SBERT and GNN assets.

## Next phase

After the API report passes review, build the professional React/TypeScript
dashboard against the versioned OpenAPI contract.
