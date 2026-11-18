import json
import statistics
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"

OUTPUT_DIR = PROJECT_ROOT / "models" / "sbert"
LOCAL_MODEL_DIR = OUTPUT_DIR / "model"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_TOKEN_LIMIT = 220
CHUNK_TOKEN_OVERLAP = 40
ENCODING_BATCH_SIZE = 64


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def clean_value(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
            str(item)
            for item in value
            if item is not None
        )

    return str(value).strip()


def build_candidate_text(profile):
    preferences = profile.get(
        "preferences",
        {},
    )

    sections = [
        (
            "Candidate roles",
            clean_value(
                profile.get("roles", [])
            ),
        ),
        (
            "Role families",
            clean_value(
                profile.get(
                    "derived_role_families",
                    [],
                )
            ),
        ),
        (
            "Primary skills",
            clean_value(
                profile.get(
                    "skills_primary",
                    [],
                )
            ),
        ),
        (
            "Secondary skills",
            clean_value(
                profile.get(
                    "skills_secondary",
                    [],
                )
            ),
        ),
        (
            "Years of experience",
            clean_value(
                profile.get(
                    "experience_years"
                )
            ),
        ),
        (
            "Seniority",
            clean_value(
                profile.get(
                    "seniority_normalized"
                )
            ),
        ),
        (
            "Industry domains",
            clean_value(
                profile.get("domains", [])
            ),
        ),
        (
            "Career objective",
            clean_value(
                profile.get("career_intent")
            ),
        ),
        (
            "Preferred locations",
            clean_value(
                preferences.get(
                    "locations",
                    [],
                )
            ),
        ),
        (
            "Remote preference",
            clean_value(
                profile.get(
                    "remote_preference"
                )
            ),
        ),
    ]

    return "\n".join(
        f"{heading}: {content}"
        for heading, content in sections
        if content
    )


def build_job_text(job):
    location_data = job.get(
        "location_normalized",
        {},
    )

    if not isinstance(location_data, dict):
        location_data = {}

    sections = [
        (
            "Job title",
            clean_value(job.get("title")),
        ),
        (
            "Role family",
            clean_value(
                job.get(
                    "derived_role_family",
                    job.get(
                        "role_family_hint"
                    ),
                )
            ),
        ),
        (
            "Seniority",
            clean_value(
                job.get(
                    "seniority_normalized",
                    job.get(
                        "seniority_hint",
                        job.get(
                            "seniority_level"
                        ),
                    ),
                )
            ),
        ),
        (
            "Minimum experience",
            clean_value(
                job.get(
                    "minimum_experience_years",
                    job.get(
                        "experience_years_hint"
                    ),
                )
            ),
        ),
        (
            "Required skills",
            clean_value(
                job.get(
                    "skills_required",
                    [],
                )
            ),
        ),
        (
            "Preferred skills",
            clean_value(
                job.get(
                    "skills_preferred",
                    [],
                )
            ),
        ),
        (
            "Other mentioned skills",
            clean_value(
                job.get(
                    "skills_mentioned",
                    [],
                )
            ),
        ),
        (
            "Industry or domain",
            clean_value(
                job.get(
                    "domain_normalized",
                    job.get(
                        "domain_hint",
                        job.get("industry"),
                    ),
                )
            ),
        ),
        (
            "Location",
            clean_value(
                job.get("location")
            ),
        ),
        (
            "Country",
            clean_value(
                location_data.get("country")
            ),
        ),
        (
            "Remote available",
            clean_value(
                job.get(
                    "remote_available",
                    job.get(
                        "remote_hint"
                    ),
                )
            ),
        ),
        (
            "Employment type",
            clean_value(
                job.get(
                    "employment_type_normalized",
                    job.get(
                        "employment_type"
                    ),
                )
            ),
        ),
        (
            "Job description",
            clean_value(
                job.get("description")
            ),
        ),
    ]

    return "\n".join(
        f"{heading}: {content}"
        for heading, content in sections
        if content
    )


def chunk_text(text, tokenizer):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not token_ids:
        return [""]

    chunk_limit = min(
        CHUNK_TOKEN_LIMIT,
        240,
    )

    step = max(
        1,
        chunk_limit - CHUNK_TOKEN_OVERLAP,
    )

    chunks = []

    for start in range(
        0,
        len(token_ids),
        step,
    ):
        chunk_ids = token_ids[
            start:start + chunk_limit
        ]

        if not chunk_ids:
            continue

        chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        chunks.append(chunk)

        if start + chunk_limit >= len(token_ids):
            break

    return chunks or [""]


def create_document_embeddings(
    document_ids,
    document_texts,
    model,
    document_type,
):
    all_chunks = []
    chunk_owners = []
    chunk_counts = []

    for document_index, text in enumerate(
        document_texts
    ):
        chunks = chunk_text(
            text,
            model.tokenizer,
        )

        chunk_counts.append(len(chunks))

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_owners.append(document_index)

    print(
        f"{document_type}: "
        f"{len(document_ids):,} documents, "
        f"{len(all_chunks):,} chunks"
    )

    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=ENCODING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_dimension = (
        chunk_embeddings.shape[1]
    )

    document_embeddings = np.zeros(
        (
            len(document_ids),
            embedding_dimension,
        ),
        dtype=np.float32,
    )

    document_chunk_totals = np.zeros(
        len(document_ids),
        dtype=np.float32,
    )

    for chunk_index, owner_index in enumerate(
        chunk_owners
    ):
        document_embeddings[owner_index] += (
            chunk_embeddings[chunk_index]
        )

        document_chunk_totals[owner_index] += 1.0

    document_embeddings /= (
        document_chunk_totals[:, None]
    )

    norms = np.linalg.norm(
        document_embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    document_embeddings = (
        document_embeddings / norms
    )

    statistics_data = {
        "document_count": len(document_ids),
        "chunk_count": len(all_chunks),
        "minimum_chunks_per_document":
            min(chunk_counts),
        "maximum_chunks_per_document":
            max(chunk_counts),
        "mean_chunks_per_document": round(
            statistics.mean(chunk_counts),
            4,
        ),
        "median_chunks_per_document": round(
            statistics.median(chunk_counts),
            4,
        ),
        "embedding_dimension":
            embedding_dimension,
    }

    return (
        document_embeddings,
        statistics_data,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles = load_json(
        PROCESSED_DIR
        / "profiles_features_enriched.json"
    )

    enriched_jobs = load_json(
        PROCESSED_DIR
        / "jobs_features_enriched.json"
    )

    raw_jobs = load_json(
        RAW_DIR / "scraped_jobs.json"
    )

    benchmark_jobs = load_json(
        PROCESSED_DIR
        / "benchmark_jobs.json"
    )

    gold_labels = load_json(
        SPLIT_DIR / "gold_evaluation.json"
    )

    profile_ids = [
        profile["profile_id"]
        for profile in profiles
    ]

    profile_texts = [
        build_candidate_text(profile)
        for profile in profiles
    ]

    jobs_by_id = {
        job["id"]: job
        for job in enriched_jobs
    }

    # Add the excluded low-quality main job for complete gold coverage.
    gold_job_ids = {
        label["job_id"]
        for label in gold_labels
    }

    for raw_job in raw_jobs:
        job_id = raw_job["id"]

        if (
            job_id in gold_job_ids
            and job_id not in jobs_by_id
        ):
            jobs_by_id[job_id] = raw_job

    # Add all standalone benchmark jobs.
    for benchmark_job in benchmark_jobs:
        jobs_by_id[
            benchmark_job["id"]
        ] = benchmark_job

    job_ids = sorted(jobs_by_id)

    job_texts = [
        build_job_text(
            jobs_by_id[job_id]
        )
        for job_id in job_ids
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Loading SBERT on: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    model.max_seq_length = 256

    print("Creating candidate embeddings...")

    (
        candidate_embeddings,
        candidate_statistics,
    ) = create_document_embeddings(
        profile_ids,
        profile_texts,
        model,
        "Candidates",
    )

    print("Creating job embeddings...")

    (
        job_embeddings,
        job_statistics,
    ) = create_document_embeddings(
        job_ids,
        job_texts,
        model,
        "Jobs",
    )

    np.save(
        OUTPUT_DIR / "candidate_embeddings.npy",
        candidate_embeddings,
    )

    np.save(
        OUTPUT_DIR / "job_embeddings.npy",
        job_embeddings,
    )

    save_json(
        OUTPUT_DIR / "candidate_ids.json",
        profile_ids,
    )

    save_json(
        OUTPUT_DIR / "job_ids.json",
        job_ids,
    )

    # Save a local model copy for later offline use by Streamlit.
    model.save(
        str(LOCAL_MODEL_DIR)
    )

    covered_job_ids = set(job_ids)

    missing_gold_job_ids = sorted(
        gold_job_ids - covered_job_ids
    )

    manifest = {
        "model": {
            "name": MODEL_NAME,
            "local_model_directory":
                str(LOCAL_MODEL_DIR),
            "device_used": device,
            "maximum_sequence_length":
                model.max_seq_length,
            "chunk_token_limit":
                CHUNK_TOKEN_LIMIT,
            "chunk_token_overlap":
                CHUNK_TOKEN_OVERLAP,
            "normalised_embeddings": True,
            "aggregation":
                "mean of overlapping chunk embeddings, "
                "followed by L2 normalisation",
        },
        "candidates": candidate_statistics,
        "jobs": job_statistics,
        "coverage": {
            "profile_count":
                len(profile_ids),
            "main_enriched_job_count":
                len(enriched_jobs),
            "total_embedded_job_count":
                len(job_ids),
            "gold_job_count":
                len(gold_job_ids),
            "gold_jobs_covered":
                len(
                    gold_job_ids
                    & covered_job_ids
                ),
            "missing_gold_job_ids":
                missing_gold_job_ids,
        },
        "files": {
            "candidate_embeddings":
                "candidate_embeddings.npy",
            "job_embeddings":
                "job_embeddings.npy",
            "candidate_ids":
                "candidate_ids.json",
            "job_ids":
                "job_ids.json",
            "local_model":
                "model/",
        },
        "safety_rules": {
            "training_labels_used":
                False,
            "validation_labels_used":
                False,
            "test_labels_used":
                False,
            "gold_labels_used_as_features":
                False,
            "long_texts_truncated_without_chunking":
                False,
            "model_saved_for_local_reuse":
                True,
        },
    }

    save_json(
        OUTPUT_DIR / "embedding_manifest.json",
        manifest,
    )

    print("")
    print(
        "Stage 7A SBERT embedding generation completed."
    )

    print(
        f"Candidate embeddings: "
        f"{candidate_embeddings.shape}"
    )

    print(
        f"Job embeddings: "
        f"{job_embeddings.shape}"
    )

    print(
        "Gold jobs covered: "
        f"{manifest['coverage']['gold_jobs_covered']}/"
        f"{manifest['coverage']['gold_job_count']}"
    )

    print("")
    print(
        "Manifest created at: "
        f"{OUTPUT_DIR / 'embedding_manifest.json'}"
    )


if __name__ == "__main__":
    main()