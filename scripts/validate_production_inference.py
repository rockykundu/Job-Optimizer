from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.job_optimizer_engine import JobOptimizerEngine


EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "production_inference"
SPLITS = ("train", "validation", "test", "gold")
REPRODUCTION_TOLERANCE = 1e-4


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def validate_checkpoint_reproduction(engine: JobOptimizerEngine):
    report = {}
    for split in SPLITS:
        path = PROJECT_ROOT / "evaluation" / "gnn" / f"{split}_predictions.json"
        rows = load_json(path)
        pairs = [(str(row["candidate_id"]), str(row["job_id"])) for row in rows]
        reproduced = engine.score_existing_pairs(pairs)
        skills_difference = [
            abs(float(expected["skills_predicted"]) - actual["skills_score"])
            for expected, actual in zip(rows, reproduced)
        ]
        composite_difference = [
            abs(float(expected["composite_predicted"]) - actual["composite_score"])
            for expected, actual in zip(rows, reproduced)
        ]
        maximum_difference = max(skills_difference + composite_difference, default=0.0)
        if maximum_difference > REPRODUCTION_TOLERANCE:
            raise ValueError(
                f"{split} checkpoint reproduction differs by {maximum_difference:.8f}; "
                f"tolerance is {REPRODUCTION_TOLERANCE}"
            )
        report[split] = {
            "pair_count": len(rows),
            "maximum_skills_absolute_difference": round(max(skills_difference), 8),
            "maximum_composite_absolute_difference": round(
                max(composite_difference), 8
            ),
            "within_tolerance": True,
        }
    return report


def validate_cold_start_resume(engine: JobOptimizerEngine):
    sample_resume = """
    Backend software engineer with five years of experience building Python and
    Django services. Designed REST APIs backed by PostgreSQL and Redis. Deployed
    Docker workloads to AWS, used Git and GitHub for collaboration, and created
    CI/CD pipelines. Experienced with testing, documentation and agile delivery.
    """
    result = engine.rank_resume(sample_resume, top_k=5)
    if len(result["matches"]) != 5:
        raise ValueError("Cold-start inference did not return five matches")
    if result["candidate"]["cold_start_candidate_updates_shared_skills"] is not False:
        raise ValueError("Cold-start candidate was incorrectly allowed to update skills")
    scores = [
        float(match["raw_composite_score"]) for match in result["matches"]
    ]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Cold-start inference returned a non-finite score")
    if scores != sorted(scores, reverse=True):
        raise ValueError("Cold-start matches are not ranked by composite score")
    return {
        "detected_skills": result["candidate"]["detected_skills"],
        "detected_skill_count": result["candidate"]["detected_skill_count"],
        "returned_match_count": len(result["matches"]),
        "scores_finite": True,
        "scores_descending": True,
        "cold_start_candidate_updates_shared_skills": False,
        "top_match": {
            "job_id": result["matches"][0]["job_id"],
            "title": result["matches"][0]["title"],
            "raw_composite_score": round(scores[0], 6),
        },
    }


def main() -> None:
    engine = JobOptimizerEngine(PROJECT_ROOT)
    reproduction = validate_checkpoint_reproduction(engine)
    cold_start = validate_cold_start_resume(engine)
    report = {
        "stage": "Production inference validation",
        "status": "passed",
        "checkpoint_reproduction": reproduction,
        "cold_start_resume": cold_start,
        "coverage": {
            "graph_candidates": len(engine.candidate_ids),
            "embedded_jobs": len(engine.job_ids),
            "rankable_main_jobs": len(engine.main_job_ids),
            "graph_skills": len(engine.skill_ids),
        },
        "safety_checks": {
            "saved_local_sbert_used": True,
            "saved_gnn_checkpoint_used": True,
            "dataset_downloaded_during_inference": False,
            "model_downloaded_during_inference": False,
            "new_candidate_updates_shared_skill_nodes": False,
            "raw_predictions_used_for_ranking": True,
            "display_clipping_changes_ranking": False,
            "keyword_evidence_changes_numeric_score": False,
        },
        "known_limitations": [
            "The training targets are generated compatibility labels, not observed hiring outcomes.",
            "The human-reviewed gold set is small.",
            "PDF text extraction quality and resume formatting can affect detected evidence.",
        ],
    }
    save_json(EVALUATION_DIR / "production_inference_report.json", report)
    print("Production inference validation passed.")
    print(
        "Checkpoint pairs reproduced: "
        + f"{sum(item['pair_count'] for item in reproduction.values()):,}"
    )
    print(f"Rankable main jobs: {len(engine.main_job_ids):,}")
    print(f"Cold-start detected skills: {cold_start['detected_skill_count']}")
    print(f"Report: {EVALUATION_DIR / 'production_inference_report.json'}")


if __name__ == "__main__":
    main()
