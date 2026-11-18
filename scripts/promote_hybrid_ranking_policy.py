from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_POLICY = ROOT / "models" / "candidate" / "hybrid_ranking_policy.json"
FINAL_POLICY = ROOT / "models" / "final" / "ranking_policy.json"
MODEL_CONFIG = ROOT / "models" / "final" / "model_config.json"
SELECTION_REPORT = (
    ROOT
    / "evaluation"
    / "ranking_policy"
    / "hybrid_ranking_selection_report.json"
)
CONTROLLED_REPORT = (
    ROOT
    / "evaluation"
    / "ranking_policy"
    / "controlled_profile_comparison.json"
)
PROMOTION_REPORT = (
    ROOT / "evaluation" / "ranking_policy" / "production_policy_promotion.json"
)
BACKUP_DIR = ROOT / "models" / "final" / "backups"

APPROVED_GNN_WEIGHT = 0.78
APPROVED_SBERT_WEIGHT = 0.22


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


def require_files() -> None:
    required = [
        CANDIDATE_POLICY,
        MODEL_CONFIG,
        SELECTION_REPORT,
        CONTROLLED_REPORT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Promotion prerequisites are missing:\n" + "\n".join(missing))


def validate_approved_evidence(
    candidate: dict[str, Any],
    selection: dict[str, Any],
    controlled: dict[str, Any],
) -> None:
    weights = candidate.get("weights", {})
    if not math.isclose(
        float(weights.get("gnn", -1)), APPROVED_GNN_WEIGHT, abs_tol=1e-9
    ) or not math.isclose(
        float(weights.get("sbert", -1)), APPROVED_SBERT_WEIGHT, abs_tol=1e-9
    ):
        raise ValueError(f"Candidate policy weights are not the approved {APPROVED_GNN_WEIGHT}/{APPROVED_SBERT_WEIGHT}")
    if candidate.get("status") != "candidate_not_promoted":
        raise ValueError("Candidate policy has an unexpected status")
    if not candidate.get("selection", {}).get("accepted_by_validation_guardrails"):
        raise ValueError("Candidate policy did not pass validation guardrails")
    if not selection.get("selection", {}).get("accepted_by_validation_guardrails"):
        raise ValueError("Selection report does not approve the candidate policy")
    if selection.get("promotion_status") != "not_promoted_pending_controlled_profile_review":
        raise ValueError("Selection report has an unexpected promotion state")

    reported_weights = selection["selection"]["weights"]
    if reported_weights != weights:
        raise ValueError("Candidate policy and selection-report weights differ")

    aggregate = controlled.get("aggregate", {})
    if int(aggregate.get("top_five_expected_hit_change", -999)) < 0:
        raise ValueError("Controlled top-five job-family relevance regressed")
    if int(aggregate.get("top_ten_expected_hit_change", 0)) <= 0:
        raise ValueError("Controlled top-ten job-family relevance did not improve")
    if int(aggregate.get("candidate_hybrid_unresolved_html_entity_count", -1)) != 0:
        raise ValueError("HTML entity defects remain in the candidate policy results")
    if controlled.get("safety", {}).get("production_checkpoint_unchanged") is not True:
        raise ValueError("Controlled report does not confirm checkpoint safety")


def backup_if_present(path: Path, timestamp: str) -> str | None:
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"{path.stem}.{timestamp}{path.suffix}"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup {backup}")
    shutil.copy2(path, backup)
    return str(backup.relative_to(ROOT))


def main() -> None:
    require_files()
    candidate = load_json(CANDIDATE_POLICY)
    selection = load_json(SELECTION_REPORT)
    controlled = load_json(CONTROLLED_REPORT)
    validate_approved_evidence(candidate, selection, controlled)

    promoted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups = {
        "previous_ranking_policy": backup_if_present(FINAL_POLICY, timestamp),
        "previous_model_config": backup_if_present(MODEL_CONFIG, timestamp),
    }

    production_policy = {
        **candidate,
        "version": "1.1.0",
        "status": "production",
        "promoted_at_utc": promoted_at,
        "promotion_evidence": {
            "selection_report": str(SELECTION_REPORT.relative_to(ROOT)),
            "controlled_profile_report": str(CONTROLLED_REPORT.relative_to(ROOT)),
            "validation_ndcg_at_10_gain": selection["selection"]["ndcg_at_10_gain"],
            "test_ndcg_at_10_gain": round(
                selection["split_evaluations"]["test"]["candidate_hybrid"]["ndcg_at_10"]
                - selection["split_evaluations"]["test"]["gnn"]["ndcg_at_10"],
                6,
            ),
            "controlled_top_ten_expected_hit_change": controlled["aggregate"][
                "top_ten_expected_hit_change"
            ],
        },
    }
    atomic_save_json(FINAL_POLICY, production_policy)

    model_config = load_json(MODEL_CONFIG)
    model_config["version"] = "1.1.0"
    model_config["production_ranking_policy"] = {
        "type": "validation_selected_gnn_sbert_hybrid",
        "path": str(FINAL_POLICY.relative_to(ROOT)),
        "weights": production_policy["weights"],
        "selection_objective": "candidate-centred composite NDCG@10",
        "ranking_and_display_use_same_score": True,
    }
    atomic_save_json(MODEL_CONFIG, model_config)

    promotion_report = {
        "stage": "Stage 11B production ranking policy promotion",
        "status": "promoted_pending_runtime_validation",
        "version": "1.1.0",
        "promoted_at_utc": promoted_at,
        "production_policy_path": str(FINAL_POLICY.relative_to(ROOT)),
        "weights": production_policy["weights"],
        "backups": backups,
        "hashes": {
            "candidate_policy_sha256": sha256(CANDIDATE_POLICY),
            "production_policy_sha256": sha256(FINAL_POLICY),
            "model_config_sha256": sha256(MODEL_CONFIG),
        },
        "safety": {
            "checkpoint_modified": False,
            "training_run": False,
            "dataset_download_requested": False,
            "model_download_requested": False,
            "rollback_metadata_recorded": True,
        },
    }
    atomic_save_json(PROMOTION_REPORT, promotion_report)

    print("Hybrid ranking policy promoted to production configuration.")
    print("Version: 1.1.0")
    print("Weights: GNN=0.78, SBERT=0.22")
    print("Runtime validation is still required.")
    print(f"Report: {PROMOTION_REPORT}")


if __name__ == "__main__":
    main()
