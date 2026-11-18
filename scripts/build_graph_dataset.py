from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "graph"

CANDIDATE_WEIGHTS = {"primary": 1.0, "secondary": 0.75}
JOB_WEIGHTS = {"required": 1.0, "preferred": 0.60, "mentioned": 0.25}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def locate(filename: str, preferred: Iterable[str] = ()) -> Path:
    matches = [p for p in ROOT.rglob(filename) if OUTPUT not in p.parents]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} anywhere below {ROOT}")
    preferred = tuple(x.lower() for x in preferred)
    matches.sort(
        key=lambda p: (
            -sum(token in str(p).lower() for token in preferred),
            len(p.parts),
            str(p).lower(),
        )
    )
    return matches[0]


def records(document: Any, expected_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [x for x in document if isinstance(x, dict)]
    if not isinstance(document, dict):
        raise TypeError("Expected a JSON list or object")
    for key in expected_keys + ("records", "data", "items"):
        value = document.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    # Also support {id: record} JSON maps.
    if document and all(isinstance(v, dict) for v in document.values()):
        result = []
        for map_id, value in document.items():
            row = dict(value)
            row.setdefault("id", map_id)
            result.append(row)
        return result
    raise ValueError(f"Could not find a record list. Top-level keys: {list(document)[:20]}")


def first_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def entity_id(row: dict[str, Any], kind: str) -> str:
    keys = (
        ("candidate_id", "profile_id", "candidateId", "profileId", "id")
        if kind == "candidate"
        else ("job_id", "jobId", "id")
    )
    value = first_value(row, keys)
    if value is None:
        raise KeyError(f"Missing {kind} ID in record with keys {list(row)[:30]}")
    return str(value)


def skill_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(skill_strings(item))
        return result
    if isinstance(value, dict):
        for key in (
            "canonical_skill",
            "canonical_name",
            "canonical",
            "normalized_skill",
            "normalized_name",
            "skill_name",
            "name",
            "skill",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return [item.strip()]
        result: list[str] = []
        for item in value.values():
            if isinstance(item, (list, dict)):
                result.extend(skill_strings(item))
        return result
    return []


def nested_container(row: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    for name in names:
        value = row.get(name)
        if isinstance(value, dict):
            return value
    return {}


def category_skills(row: dict[str, Any], kind: str, category: str) -> list[str]:
    if kind == "candidate":
        containers = (
            row,
            nested_container(
                row,
                (
                    "normalized_skills",
                    "skills_normalized",
                    "canonical_skills",
                    "skill_groups",
                ),
            ),
        )
        aliases = {
            "primary": (
                "skills_primary",
                "primary_skills_normalized",
                "normalized_primary_skills",
                "primary_skills",
                "primary",
            ),
            "secondary": (
                "skills_secondary",
                "secondary_skills_normalized",
                "normalized_secondary_skills",
                "secondary_skills",
                "secondary",
            ),
        }[category]
    else:
        containers = (
            row,
            nested_container(
                row,
                (
                    "extracted_skills",
                    "job_skills",
                    "normalized_skills",
                    "skills_by_requirement",
                ),
            ),
        )
        aliases = {
            "required": ("skills_required", "required_skills", "required"),
            "preferred": (
                "skills_preferred",
                "preferred_skills",
                "preferred",
                "nice_to_have",
            ),
            "mentioned": (
                "skills_mentioned",
                "mentioned_skills",
                "mentioned",
                "general_skills",
                "general",
            ),
        }[category]

    for container in containers:
        for key in aliases:
            if key in container:
                return skill_strings(container[key])
    return []


def catalogue_names(document: Any) -> list[str]:
    values: list[Any]
    if isinstance(document, list):
        values = document
    elif isinstance(document, dict):
        for key in ("skills", "catalog", "canonical_skills", "items"):
            if isinstance(document.get(key), list):
                values = document[key]
                break
        else:
            values = list(document.values())
    else:
        values = []
    result: list[str] = []
    for value in values:
        result.extend(skill_strings(value))
    return result


def locate_split(split: str) -> Path:
    split_dir = PROCESSED / "splits"
    candidates = []
    if split == "gold":
        # This is the held-out 108-pair human-reviewed evaluation set.
        # Never substitute reserved_gold_candidate_generated_labels.json,
        # which contains generated labels excluded from model training.
        candidates.extend(
            [
                split_dir / "gold_evaluation.json",
                split_dir / "gold_labels.json",
            ]
        )
    candidates.extend([
        split_dir / f"{split}_labels.json",
        split_dir / f"{split}.json",
        split_dir / f"labels_{split}.json",
        split_dir / f"{split}_pairs.json",
    ])
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(split_dir.glob(f"*{split}*.json")) if split_dir.exists() else []
    matches = [
        p
        for p in matches
        if "manifest" not in p.name.lower()
        and "report" not in p.name.lower()
        and "reserved" not in p.name.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not uniquely locate the {split} label file in {split_dir}. "
        f"Candidates found: {[p.name for p in matches]}"
    )


def numeric_target(row: dict[str, Any], target: str) -> float:
    aliases = {
        "skills": ("skills_score", "skill_score", "skillsScore", "skill_match_score"),
        "composite": (
            "composite_score",
            "overall_score",
            "compatibility_score",
            "compositeScore",
            "score",
        ),
    }[target]
    containers = [row]
    for name in ("scores", "label", "labels", "targets"):
        if isinstance(row.get(name), dict):
            containers.append(row[name])
    for container in containers:
        value = first_value(container, aliases)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    raise KeyError(f"Missing numeric {target} target in label keys {list(row)[:30]}")


def canonicalise(raw: str, spelling: dict[str, str]) -> str:
    cleaned = " ".join(raw.strip().split())
    key = cleaned.casefold()
    spelling.setdefault(key, cleaned)
    return spelling[key]


def build_relation_edges(
    rows: list[dict[str, Any]],
    kind: str,
    node_index: dict[str, int],
    categories: dict[str, float],
    spelling: dict[str, str],
) -> tuple[dict[tuple[int, str], tuple[float, int]], Counter[str], set[str]]:
    edges: dict[tuple[int, str], tuple[float, int]] = {}
    counts: Counter[str] = Counter()
    nodes_with_skills: set[str] = set()
    for row in rows:
        node_id = entity_id(row, kind)
        if node_id not in node_index:
            continue
        for category_code, (category, weight) in enumerate(categories.items()):
            unique = {
                canonicalise(skill, spelling)
                for skill in category_skills(row, kind, category)
                if skill.strip()
            }
            if unique:
                nodes_with_skills.add(node_id)
            for skill in unique:
                key = (node_index[node_id], skill)
                old = edges.get(key)
                if old is None or weight > old[0]:
                    edges[key] = (weight, category_code)
                counts[category] += 1
    return edges, counts, nodes_with_skills


def materialise_edges(
    edges: dict[tuple[int, str], tuple[float, int]], skill_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(edges.items(), key=lambda x: (x[0][0], skill_index[x[0][1]]))
    edge_index = np.asarray(
        [[node, skill_index[skill]] for (node, skill), _ in ordered], dtype=np.int64
    ).T
    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    weights = np.asarray([meta[0] for _, meta in ordered], dtype=np.float32)
    categories = np.asarray([meta[1] for _, meta in ordered], dtype=np.int8)
    return edge_index, weights, categories


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    profile_path = locate("profiles_skills_normalized.json", ("processed",))
    job_path = locate("jobs_skills_extracted.json", ("processed",))
    catalog_path = locate("skill_catalog.json", ("processed",))
    candidate_embedding_path = locate("candidate_embeddings.npy", ("sbert", "processed"))
    job_embedding_path = locate("job_embeddings.npy", ("sbert", "processed"))
    candidate_ids_path = locate("candidate_ids.json", ("sbert", "processed"))
    job_ids_path = locate("job_ids.json", ("sbert", "processed"))
    local_model_path = ROOT / "models" / "sbert" / "model"
    if not local_model_path.exists():
        local_model_path = locate("config.json", ("models", "sbert", "model")).parent

    # Resolve every supervision file before doing any model inference. This
    # makes path/schema failures fail fast instead of wasting encoding time.
    split_paths = {
        split: locate_split(split)
        for split in ("train", "validation", "test", "gold")
    }

    profiles = records(load_json(profile_path), ("profiles", "candidates"))
    jobs = records(load_json(job_path), ("jobs",))
    candidate_ids = [str(x) for x in load_json(candidate_ids_path)]
    job_ids = [str(x) for x in load_json(job_ids_path)]
    candidate_features = np.load(candidate_embedding_path).astype(np.float32)
    job_features = np.load(job_embedding_path).astype(np.float32)

    if candidate_features.shape[0] != len(candidate_ids):
        raise ValueError("Candidate embedding rows do not match candidate_ids.json")
    if job_features.shape[0] != len(job_ids):
        raise ValueError("Job embedding rows do not match job_ids.json")
    if candidate_features.shape[1] != job_features.shape[1]:
        raise ValueError("Candidate and job embedding dimensions differ")

    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    job_index = {value: index for index, value in enumerate(job_ids)}
    if len(candidate_index) != len(candidate_ids) or len(job_index) != len(job_ids):
        raise ValueError("Duplicate IDs detected in an embedding ID file")

    spelling: dict[str, str] = {}
    catalog_skill_names = catalogue_names(load_json(catalog_path))
    catalog_skill_keys = {
        " ".join(skill.strip().split()).casefold()
        for skill in catalog_skill_names
        if skill.strip()
    }
    for skill in catalog_skill_names:
        canonicalise(skill, spelling)

    candidate_edges, candidate_category_counts, candidates_with_skills = build_relation_edges(
        profiles, "candidate", candidate_index, CANDIDATE_WEIGHTS, spelling
    )
    job_edges, job_category_counts, jobs_with_skills = build_relation_edges(
        jobs, "job", job_index, JOB_WEIGHTS, spelling
    )

    connected_skill_keys = {
        skill.casefold()
        for _, skill in list(candidate_edges) + list(job_edges)
    }
    connected_skills_not_in_catalog = sorted(
        (spelling[key] for key in connected_skill_keys - catalog_skill_keys),
        key=lambda value: (value.casefold(), value),
    )

    if len(candidates_with_skills) < int(0.90 * len(candidate_ids)):
        raise ValueError(
            f"Only {len(candidates_with_skills)}/{len(candidate_ids)} candidates produced skill edges. "
            "Stop: the profile skill schema was not recognised safely."
        )
    main_job_ids = {entity_id(row, "job") for row in jobs}
    if len(jobs_with_skills) < int(0.90 * len(main_job_ids)):
        raise ValueError(
            f"Only {len(jobs_with_skills)}/{len(main_job_ids)} main jobs produced skill edges. "
            "Stop: the job skill schema was not recognised safely."
        )

    skills = sorted(spelling.values(), key=lambda x: (x.casefold(), x))
    skill_index = {value: index for index, value in enumerate(skills)}
    candidate_edge_index, candidate_edge_weight, candidate_edge_category = materialise_edges(
        candidate_edges, skill_index
    )
    job_edge_index, job_edge_weight, job_edge_category = materialise_edges(job_edges, skill_index)

    print(f"Encoding {len(skills):,} canonical skill names with the saved local SBERT model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(local_model_path), device="cpu")
    skill_features = model.encode(
        [f"Technical or professional skill: {skill}" for skill in skills],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if skill_features.shape[1] != candidate_features.shape[1]:
        raise ValueError("Skill embedding dimension differs from candidate/job embeddings")

    split_candidate_sets: dict[str, set[str]] = {}
    supervision: dict[str, dict[str, np.ndarray]] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    for split in ("train", "validation", "test", "gold"):
        path = split_paths[split]
        label_rows = records(load_json(path), ("labels", "pairs"))
        c_indices: list[int] = []
        j_indices: list[int] = []
        y_skills: list[float] = []
        y_composite: list[float] = []
        pairs: set[tuple[str, str]] = set()
        for row in label_rows:
            candidate_id = entity_id(row, "candidate")
            job_id = entity_id(row, "job")
            if candidate_id not in candidate_index:
                raise KeyError(f"{split}: candidate {candidate_id} has no SBERT embedding")
            if job_id not in job_index:
                raise KeyError(f"{split}: job {job_id} has no SBERT embedding")
            pair = (candidate_id, job_id)
            if pair in pairs:
                raise ValueError(f"{split}: duplicate labelled pair {pair}")
            pairs.add(pair)
            c_indices.append(candidate_index[candidate_id])
            j_indices.append(job_index[job_id])
            y_skills.append(numeric_target(row, "skills"))
            y_composite.append(numeric_target(row, "composite"))
        supervision[split] = {
            "candidate_index": np.asarray(c_indices, dtype=np.int64),
            "job_index": np.asarray(j_indices, dtype=np.int64),
            "skills_score": np.asarray(y_skills, dtype=np.float32),
            "composite_score": np.asarray(y_composite, dtype=np.float32),
        }
        split_candidate_sets[split] = {candidate_ids[i] for i in set(c_indices)}
        pair_sets[split] = pairs

    split_names = tuple(split_candidate_sets)
    candidate_overlaps: dict[str, int] = {}
    pair_overlaps: dict[str, int] = {}
    for left_position, left in enumerate(split_names):
        for right in split_names[left_position + 1 :]:
            key = f"{left}_{right}"
            candidate_overlaps[key] = len(split_candidate_sets[left] & split_candidate_sets[right])
            pair_overlaps[key] = len(pair_sets[left] & pair_sets[right])
    if any(candidate_overlaps.values()) or any(pair_overlaps.values()):
        raise ValueError(
            f"Leakage detected. Candidate overlaps={candidate_overlaps}; pair overlaps={pair_overlaps}"
        )

    candidate_split = np.full(len(candidate_ids), -1, dtype=np.int8)
    split_code = {"train": 0, "validation": 1, "test": 2, "gold": 3}
    for split, ids in split_candidate_sets.items():
        for candidate_id in ids:
            candidate_split[candidate_index[candidate_id]] = split_code[split]

    np.savez_compressed(
        OUTPUT / "node_features.npz",
        candidate=candidate_features,
        job=job_features,
        skill=skill_features,
    )
    np.savez_compressed(
        OUTPUT / "graph_edges.npz",
        candidate_skill_edge_index=candidate_edge_index,
        candidate_skill_weight=candidate_edge_weight,
        candidate_skill_category=candidate_edge_category,
        job_skill_edge_index=job_edge_index,
        job_skill_weight=job_edge_weight,
        job_skill_category=job_edge_category,
    )
    np.savez_compressed(OUTPUT / "candidate_splits.npz", split_code=candidate_split)
    for split, arrays in supervision.items():
        np.savez_compressed(OUTPUT / f"{split}_supervision.npz", **arrays)
    save_json(OUTPUT / "candidate_ids.json", candidate_ids)
    save_json(OUTPUT / "job_ids.json", job_ids)
    save_json(OUTPUT / "skill_ids.json", skills)

    report = {
        "stage": "Stage 8A - leakage-safe heterogeneous graph construction",
        "node_counts": {
            "candidate": len(candidate_ids),
            "job": len(job_ids),
            "skill": len(skills),
        },
        "node_feature_shapes": {
            "candidate": list(candidate_features.shape),
            "job": list(job_features.shape),
            "skill": list(skill_features.shape),
        },
        "relation_counts": {
            "candidate_to_skill": int(candidate_edge_index.shape[1]),
            "job_to_skill": int(job_edge_index.shape[1]),
            "candidate_categories_before_deduplication": dict(candidate_category_counts),
            "job_categories_before_deduplication": dict(job_category_counts),
        },
        "coverage": {
            "candidates_with_skill_edges": len(candidates_with_skills),
            "candidates_without_skill_edges": len(candidate_ids) - len(candidates_with_skills),
            "main_jobs_with_skill_edges": len(jobs_with_skills),
            "main_jobs_without_skill_edges": len(main_job_ids) - len(jobs_with_skills),
            "embedded_extra_jobs_without_extracted_skill_records": len(set(job_ids) - main_job_ids),
        },
        "catalogue_alignment": {
            "catalogue_skill_count": len(catalog_skill_keys),
            "connected_skill_count": len(connected_skill_keys),
            "connected_skills_missing_from_catalogue_count": len(
                connected_skills_not_in_catalog
            ),
            "connected_skills_missing_from_catalogue": connected_skills_not_in_catalog,
        },
        "supervision": {
            split: {
                "pair_count": int(values["candidate_index"].shape[0]),
                "candidate_count": len(split_candidate_sets[split]),
                "skills_score_mean": round(float(values["skills_score"].mean()), 4),
                "composite_score_mean": round(float(values["composite_score"].mean()), 4),
            }
            for split, values in supervision.items()
        },
        "leakage_checks": {
            "candidate_overlaps": candidate_overlaps,
            "pair_overlaps": pair_overlaps,
            "all_overlap_counts_zero": not any(candidate_overlaps.values())
            and not any(pair_overlaps.values()),
        },
        "graph_semantics": {
            "message_passing_relations": ["candidate-to-skill", "job-to-skill"],
            "candidate_job_labels_used_as_message_passing_edges": False,
            "candidate_job_labels_stored_as_separate_supervision": True,
            "candidate_skill_weights": CANDIDATE_WEIGHTS,
            "job_skill_weights": JOB_WEIGHTS,
            "skill_features": "Saved local SBERT model encoding of canonical skill names",
            "candidate_and_job_features": "Previously saved normalised SBERT embeddings",
        },
        "training_visibility_rules": {
            "train_candidate_split_code": 0,
            "validation_candidate_split_code": 1,
            "test_candidate_split_code": 2,
            "gold_candidate_split_code": 3,
            "unassigned_candidate_split_code": -1,
            "training_must_filter_candidate_nodes_to_code_zero": True,
            "validation_test_gold_labels_available_to_training": False,
            "all_job_feature_nodes_available": True,
        },
        "input_files": {
            "profiles": str(profile_path.relative_to(ROOT)),
            "jobs": str(job_path.relative_to(ROOT)),
            "skill_catalog": str(catalog_path.relative_to(ROOT)),
            "candidate_embeddings": str(candidate_embedding_path.relative_to(ROOT)),
            "job_embeddings": str(job_embedding_path.relative_to(ROOT)),
            "splits": {k: str(v.relative_to(ROOT)) for k, v in split_paths.items()},
        },
        "output_files": [
            "node_features.npz",
            "graph_edges.npz",
            "candidate_splits.npz",
            "train_supervision.npz",
            "validation_supervision.npz",
            "test_supervision.npz",
            "gold_supervision.npz",
            "candidate_ids.json",
            "job_ids.json",
            "skill_ids.json",
            "graph_manifest.json",
        ],
    }
    save_json(OUTPUT / "graph_manifest.json", report)

    print("\nStage 8A graph construction completed.")
    print(f"Candidates: {len(candidate_ids):,}")
    print(f"Jobs: {len(job_ids):,}")
    print(f"Skills: {len(skills):,}")
    print(f"Candidate-skill edges: {candidate_edge_index.shape[1]:,}")
    print(f"Job-skill edges: {job_edge_index.shape[1]:,}")
    print(f"Report: {OUTPUT / 'graph_manifest.json'}")


if __name__ == "__main__":
    main()
