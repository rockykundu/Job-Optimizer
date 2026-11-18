from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ranking_quality_scenarios import SCENARIOS
from src.job_optimizer_engine import JobOptimizerEngine


FINAL_POLICY = ROOT / "models" / "final" / "ranking_policy.json"
MODEL_CONFIG = ROOT / "models" / "final" / "model_config.json"
CONTROLLED_REPORT = (
    ROOT
    / "evaluation"
    / "ranking_policy"
    / "controlled_profile_comparison.json"
)
PROMOTION_REPORT = (
    ROOT / "evaluation" / "ranking_policy" / "production_policy_promotion.json"
)
OUTPUT_PATH = (
    ROOT
    / "evaluation"
    / "ranking_policy"
    / "production_policy_validation_report.json"
)
CHECKPOINT = ROOT / "models" / "gnn" / "heterogeneous_gnn.pt"
ENTITY_PATTERN = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.IGNORECASE)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def atomic_save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        FINAL_POLICY,
        MODEL_CONFIG,
        CONTROLLED_REPORT,
        PROMOTION_REPORT,
        CHECKPOINT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Runtime-validation prerequisites are missing:\n" + "\n".join(missing))

    policy = load_json(FINAL_POLICY)
    config = load_json(MODEL_CONFIG)
    controlled = load_json(CONTROLLED_REPORT)
    promotion = load_json(PROMOTION_REPORT)

    if policy.get("status") != "production" or policy.get("version") != "1.1.0":
        raise ValueError("Final ranking policy is not production version 1.1.0")
    if policy.get("weights") != {"gnn": 0.78, "sbert": 0.22}:
        raise ValueError("Final ranking policy does not contain the approved weights")
    configured = config.get("production_ranking_policy", {})
    if configured.get("weights") != policy["weights"]:
        raise ValueError("Model configuration and production policy weights differ")
    if promotion.get("status") != "promoted_pending_runtime_validation":
        raise ValueError("Promotion report has an unexpected pre-validation status")

    expected_by_scenario = {
        row["scenario_id"]: row["candidate_hybrid"]
        for row in controlled["comparisons"]
    }
    engine = JobOptimizerEngine(ROOT)
    if engine.ranking_policy is None:
        raise ValueError("Engine did not automatically load the production policy")

    reproduced_results = []
    reproduced_job_count = 0
    html_entity_count = 0
    formula_max_absolute_error = 0.0
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"[{index}/{len(SCENARIOS)}] Validating {scenario['id']}...")
        result = engine.rank_resume(scenario["resume"], top_k=10)
        if result["ranking_policy"] != policy["policy_name"]:
            raise ValueError("Inference response did not identify the production policy")

        expected = expected_by_scenario[scenario["id"]]["top_results"]
        actual_ids = [match["job_id"] for match in result["matches"]]
        expected_ids = [match["job_id"] for match in expected]
        if actual_ids != expected_ids:
            raise ValueError(
                f"{scenario['id']}: production top-ten IDs differ from the approved candidate policy"
            )

        score_error = 0.0
        for match, expected_match in zip(result["matches"], expected):
            expected_score = (
                policy["weights"]["gnn"] * float(match["raw_composite_score"])
                + policy["weights"]["sbert"]
                * float(match["sbert_calibrated_composite_score"])
            )
            formula_error = abs(float(match["ranking_score"]) - expected_score)
            formula_max_absolute_error = max(
                formula_max_absolute_error, formula_error
            )
            score_error = max(
                score_error,
                abs(
                    float(match["ranking_score"])
                    - float(expected_match["ranking_score"])
                ),
            )
            if ENTITY_PATTERN.search(match["title"]):
                html_entity_count += 1
        if score_error > 1e-4:
            raise ValueError(
                f"{scenario['id']}: maximum reproduced score error {score_error} exceeds tolerance"
            )
        reproduced_job_count += len(actual_ids)
        reproduced_results.append(
            {
                "scenario_id": scenario["id"],
                "top_ten_job_ids_exact": True,
                "maximum_score_absolute_error": round(score_error, 10),
                "top_job_id": actual_ids[0],
                "top_job_title": result["matches"][0]["title"],
                "top_job_score": round(
                    float(result["matches"][0]["ranking_score"]), 6
                ),
            }
        )

    if formula_max_absolute_error > 1e-8:
        raise ValueError("Production ranking formula was not reproduced exactly")
    if html_entity_count:
        raise ValueError(f"{html_entity_count} HTML entity titles remain")

    validation_report = {
        "stage": "Stage 11B production ranking policy runtime validation",
        "status": "passed",
        "version": "1.1.0",
        "ranking_policy": policy["policy_name"],
        "weights": policy["weights"],
        "rankable_job_count": len(engine.main_job_ids),
        "controlled_scenario_count": len(SCENARIOS),
        "reproduced_top_ten_job_count": reproduced_job_count,
        "all_top_ten_job_ids_reproduced_exactly": True,
        "maximum_formula_absolute_error": round(
            formula_max_absolute_error, 12
        ),
        "remaining_html_entity_title_count": html_entity_count,
        "scenario_results": reproduced_results,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "safety": {
            "checkpoint_modified_by_validation": False,
            "training_run": False,
            "dataset_download_requested": False,
            "model_download_requested": False,
            "production_policy_loaded_automatically": True,
            "ranking_and_display_use_same_hybrid_score": True,
        },
    }
    atomic_save_json(OUTPUT_PATH, validation_report)

    promotion["status"] = "promoted_and_runtime_validated"
    promotion["runtime_validation_report"] = str(OUTPUT_PATH.relative_to(ROOT))
    promotion["runtime_validation_checkpoint_sha256"] = validation_report[
        "checkpoint_sha256"
    ]
    atomic_save_json(PROMOTION_REPORT, promotion)

    print("Production hybrid ranking policy validation passed.")
    print(f"Rankable jobs: {len(engine.main_job_ids):,}")
    print(f"Controlled top-ten results reproduced: {reproduced_job_count}")
    print("Remaining HTML entity titles: 0")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
