FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Required system library for CPU-based ML packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies
COPY requirements-cloud.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && python -m pip install -r requirements-cloud.txt

# Backend application
COPY api/ ./api/
COPY src/ ./src/

# Only the Python scripts required by the inference engine
COPY scripts/build_sbert_embeddings.py ./scripts/
COPY scripts/extract_job_skills.py ./scripts/
COPY scripts/train_heterogeneous_gnn.py ./scripts/

# Model files
COPY models/sbert/model/ ./models/sbert/model/
COPY models/gnn/heterogeneous_gnn.pt ./models/gnn/
COPY models/final/model_config.json ./models/final/
COPY models/final/ranking_policy.json ./models/final/

# Required graph and processed data
COPY data/processed/graph/ ./data/processed/graph/
COPY data/processed/jobs_features_enriched.json ./data/processed/
COPY data/processed/skill_catalog.json ./data/processed/
COPY data/processed/skill_aliases.json ./data/processed/

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
