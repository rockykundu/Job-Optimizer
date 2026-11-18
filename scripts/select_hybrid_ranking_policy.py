from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GNN_DIR = ROOT / "evaluation" / "gnn"
SBERT_DIR = ROOT / "evaluation" / "sbert"
OUTPUT_DIR = ROOT / "evaluation" / "ranking_policy"
POLICY_PATH = ROOT / "models" / "candidate" / "hybrid_ranking_policy.json"

SPLITS = ("train", "validation", "test", "gold")
WEIGHT_STEP = 0.01
MINIMUM_NDCG_GAIN = 0.002
MAXIMUM_PAIRWISE_DECREASE = 0.005
MAXIMUM_MAE_INCREASE_RATIO = 0.05


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def indexed_rows(path: Path, candidate_field: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a list in {path}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row[candidate_field]), str(row["job_id"]))
        if key in result:
            raise ValueError(f"Duplicate candidate-job pair in {path}: {key}")
        result[key] = row
    return result


def load_aligned_split(split: str) -> list[dict[str, Any]]:
    gnn_path = GNN_DIR / f"{split}_predictions.json"
    sbert_path = SBERT_DIR / f"{split}_predictions.json"
    missing = [str(path) for path in (gnn_path, sbert_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required prediction files:\n" + "\n".join(missing))

    gnn = indexed_rows(gnn_path, "candidate_id")
    sbert = indexed_rows(sbert_path, "profile_id")
    if set(gnn) != set(sbert):
        raise ValueError(
            f"{split}: GNN and SBERT pair sets differ. "
            f"Only GNN={len(set(gnn) - set(sbert))}; "
            f"only SBERT={len(set(sbert) - set(gnn))}"
        )

    aligned = []
    for key, gnn_row in gnn.items():
        sbert_row = sbert[key]
        actual = float(gnn_row["composite_actual"])
        if not math.isclose(actual, float(sbert_row["composite_score"]), abs_tol=1e-8):
            raise ValueError(f"{split}: target mismatch for {key}")
        aligned.append(
            {
                "candidate_id": key[0],
                "job_id": key[1],
                "actual": actual,
                "gnn": float(gnn_row["composite_predicted"]),
                "sbert": float(sbert_row["calibrated_composite_prediction"]),
            }
        )
    return aligned


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_ids = np.asarray([row["candidate_id"] for row in rows], dtype=object)
    actual = np.asarray([row["actual"] for row in rows], dtype=np.float64)
    gnn = np.asarray([row["gnn"] for row in rows], dtype=np.float64)
    sbert = np.asarray([row["sbert"] for row in rows], dtype=np.float64)
    for name, values in (("actual", actual), ("gnn", gnn), ("sbert", sbert)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} values detected")
    return candidate_ids, actual, gnn, sbert


def groups(candidate_ids: np.ndarray) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, candidate_id in enumerate(candidate_ids.tolist()):
        grouped[str(candidate_id)].append(index)
    return list(grouped.values())


def centred_ndcg(
    candidate_ids: np.ndarray, actual: np.ndarray, predicted: np.ndarray, k: int
) -> float | None:
    scores = []
    for indices in groups(candidate_ids):
        if len(indices) < 2:
            continue
        truth = actual[indices]
        relevance = truth - truth.min()
        if relevance.max() <= 0:
            continue
        cutoff = min(k, len(indices))
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        predicted_order = np.argsort(-predicted[indices], kind="mergesort")[:cutoff]
        ideal_order = np.argsort(-relevance, kind="mergesort")[:cutoff]
        dcg = float(np.sum(relevance[predicted_order] * discounts))
        ideal = float(np.sum(relevance[ideal_order] * discounts))
        if ideal > 0:
            scores.append(dcg / ideal)
    return float(np.mean(scores)) if scores else None


def pairwise_accuracy(
    candidate_ids: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> float | None:
    correct = 0.0
    comparisons = 0
    for indices in groups(candidate_ids):
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                true_difference = actual[left] - actual[right]
                if true_difference == 0:
                    continue
                predicted_difference = predicted[left] - predicted[right]
                comparisons += 1
                if predicted_difference == 0:
                    correct += 0.5
                elif np.sign(predicted_difference) == np.sign(true_difference):
                    correct += 1.0
    return correct / comparisons if comparisons else None


def metrics(
    candidate_ids: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    return {
        "pair_count": int(len(actual)),
        "candidate_count": len(groups(candidate_ids)),
        "mae": round(float(np.mean(np.abs(predicted - actual))), 6),
        "ndcg_at_5": round_optional(centred_ndcg(candidate_ids, actual, predicted, 5)),
        "ndcg_at_10": round_optional(centred_ndcg(candidate_ids, actual, predicted, 10)),
        "pairwise_accuracy": round_optional(
            pairwise_accuracy(candidate_ids, actual, predicted)
        ),
        "prediction_minimum": round(float(predicted.min()), 6),
        "prediction_maximum": round(float(predicted.max()), 6),
        "prediction_standard_deviation": round(float(predicted.std()), 6),
    }


def round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def blend(gnn: np.ndarray, sbert: np.ndarray, gnn_weight: float) -> np.ndarray:
    return gnn_weight * gnn + (1.0 - gnn_weight) * sbert


def choose_on_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids, actual, gnn, sbert = arrays(rows)
    baseline = metrics(candidate_ids, actual, gnn)
    candidates = []
    steps = int(round(1.0 / WEIGHT_STEP))
    for units in range(steps + 1):
        gnn_weight = units / steps
        predicted = blend(gnn, sbert, gnn_weight)
        candidate_metrics = metrics(candidate_ids, actual, predicted)
        candidates.append(
            {
                "gnn_weight": gnn_weight,
                "sbert_weight": 1.0 - gnn_weight,
                "metrics": candidate_metrics,
            }
        )

    def objective(item: dict[str, Any]) -> tuple[float, float, float]:
        values = item["metrics"]
        return (
            -1.0 if values["ndcg_at_10"] is None else values["ndcg_at_10"],
            -1.0 if values["pairwise_accuracy"] is None else values["pairwise_accuracy"],
            -values["mae"],
        )

    eligible = [
        item
        for item in candidates
        if (
            item["metrics"]["pairwise_accuracy"]
            >= baseline["pairwise_accuracy"] - MAXIMUM_PAIRWISE_DECREASE
            and item["metrics"]["mae"]
            <= baseline["mae"] * (1.0 + MAXIMUM_MAE_INCREASE_RATIO)
        )
    ]
    # GNN-only is always eligible because it defines both guardrail baselines.
    best = max(eligible, key=objective)
    ndcg_gain = float(best["metrics"]["ndcg_at_10"] - baseline["ndcg_at_10"])
    pairwise_change = float(
        best["metrics"]["pairwise_accuracy"] - baseline["pairwise_accuracy"]
    )
    mae_change_ratio = float(
        (best["metrics"]["mae"] - baseline["mae"]) / baseline["mae"]
    )
    accepted = ndcg_gain >= MINIMUM_NDCG_GAIN
    return {
        "weights": {
            "gnn": round(float(best["gnn_weight"]), 6),
            "sbert": round(float(best["sbert_weight"]), 6),
        },
        "accepted_by_validation_guardrails": accepted,
        "baseline_gnn_metrics": baseline,
        "candidate_hybrid_metrics": best["metrics"],
        "ndcg_at_10_gain": round(ndcg_gain, 6),
        "pairwise_accuracy_change": round(pairwise_change, 6),
        "mae_change_ratio": round(mae_change_ratio, 6),
        "eligible_weight_count": len(eligible),
        "evaluated_weight_count": len(candidates),
    }


def evaluate_split(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    candidate_ids, actual, gnn, sbert = arrays(rows)
    hybrid = blend(gnn, sbert, float(weights["gnn"]))
    return {
        "gnn": metrics(candidate_ids, actual, gnn),
        "sbert": metrics(candidate_ids, actual, sbert),
        "candidate_hybrid": metrics(candidate_ids, actual, hybrid),
    }


def main() -> None:
    sbert_report_path = SBERT_DIR / "sbert_report.json"
    if not sbert_report_path.exists():
        raise FileNotFoundError(f"Missing {sbert_report_path}")

    # Selection is completed before test and gold predictions are loaded.
    validation_rows = load_aligned_split("validation")
    selection = choose_on_validation(validation_rows)
    frozen_weights = selection["weights"]

    split_rows = {"validation": validation_rows}
    for split in ("train", "test", "gold"):
        split_rows[split] = load_aligned_split(split)

    evaluations = {
        split: evaluate_split(split_rows[split], frozen_weights) for split in SPLITS
    }
    sbert_report = load_json(sbert_report_path)
    calibrator = sbert_report["calibrators"]["composite_target"]

    policy = {
        "policy_name": "validation_selected_gnn_sbert_hybrid",
        "version": "1.0.0-candidate",
        "status": "candidate_not_promoted",
        "weights": frozen_weights,
        "sbert_composite_calibrator": {
            "slope": float(calibrator["slope"]),
            "intercept": float(calibrator["intercept"]),
            "fitted_on": "training_labels_only",
        },
        "selection": {
            "split": "validation",
            "primary_objective": "candidate-centred composite NDCG@10",
            "secondary_objective": "pairwise ranking accuracy",
            "guardrails": {
                "minimum_ndcg_at_10_gain": MINIMUM_NDCG_GAIN,
                "maximum_pairwise_accuracy_decrease": MAXIMUM_PAIRWISE_DECREASE,
                "maximum_mae_increase_ratio": MAXIMUM_MAE_INCREASE_RATIO,
            },
            "accepted_by_validation_guardrails": selection[
                "accepted_by_validation_guardrails"
            ],
        },
        "score_policy": {
            "formula": "gnn_weight * gnn_composite + sbert_weight * calibrated_sbert_composite",
            "ranking_and_display_use_same_score": True,
            "artificial_offset": False,
            "manual_score_cap": False,
        },
    }
    save_json(POLICY_PATH, policy)

    report = {
        "stage": "Stage 11 candidate hybrid ranking selection",
        "status": "candidate_policy_created",
        "promotion_status": "not_promoted_pending_controlled_profile_review",
        "selection": selection,
        "split_evaluations": evaluations,
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "leakage_controls": {
            "weights_selected_from_validation_only": True,
            "test_loaded_after_weights_frozen": True,
            "gold_loaded_after_weights_frozen": True,
            "test_used_for_weight_selection": False,
            "gold_used_for_weight_selection": False,
            "checkpoint_retrained": False,
        },
    }
    report_path = OUTPUT_DIR / "hybrid_ranking_selection_report.json"
    save_json(report_path, report)

    print("Stage 11 candidate ranking policy created.")
    print(
        f"Weights: GNN={frozen_weights['gnn']:.2f}, "
        f"SBERT={frozen_weights['sbert']:.2f}"
    )
    print(
        "Validation guardrails: "
        + (
            "passed"
            if selection["accepted_by_validation_guardrails"]
            else "not passed; GNN remains production policy"
        )
    )
    print(f"Report: {report_path}")
    print(f"Candidate policy: {POLICY_PATH}")


if __name__ == "__main__":
    main()
