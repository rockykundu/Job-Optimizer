from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KEYWORD_DIR = ROOT / "evaluation" / "keyword_baseline"
SBERT_DIR = ROOT / "evaluation" / "sbert"
GNN_DIR = ROOT / "evaluation" / "gnn"
OUTPUT_DIR = ROOT / "evaluation" / "final_model"
MODEL_DIR = ROOT / "models" / "final"

SPLITS = ("train", "validation", "test", "gold")
TARGETS = ("skills", "composite")
WEIGHT_STEP = 0.01
MINIMUM_RELATIVE_VALIDATION_IMPROVEMENT = 0.01


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def required_paths(split: str) -> dict[str, Path]:
    return {
        "keyword": KEYWORD_DIR / f"{split}_predictions_calibrated.json",
        "sbert": SBERT_DIR / f"{split}_predictions.json",
        "gnn": GNN_DIR / f"{split}_predictions.json",
    }


def pair_key(model: str, row: dict[str, Any]) -> tuple[str, str]:
    candidate_field = "candidate_id" if model == "gnn" else "profile_id"
    return str(row[candidate_field]), str(row["job_id"])


def actual_value(model: str, row: dict[str, Any], target: str) -> float:
    field = f"{target}_actual" if model == "gnn" else f"{target}_score"
    return float(row[field])


def prediction_value(model: str, row: dict[str, Any], target: str) -> float | None:
    if model == "gnn":
        value = row.get(f"{target}_predicted")
    else:
        value = row.get(f"calibrated_{target}_prediction")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def indexed_rows(model: str, path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a list in {path}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = pair_key(model, row)
        if key in result:
            raise ValueError(f"Duplicate {model} pair {key} in {path}")
        result[key] = row
    return result


def load_aligned_split(split: str) -> list[dict[str, Any]]:
    paths = required_paths(split)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation files:\n" + "\n".join(missing))

    model_rows = {model: indexed_rows(model, path) for model, path in paths.items()}
    reference_keys = set(model_rows["gnn"])
    for model in ("keyword", "sbert"):
        if set(model_rows[model]) != reference_keys:
            missing_from_model = sorted(reference_keys - set(model_rows[model]))[:10]
            extra_in_model = sorted(set(model_rows[model]) - reference_keys)[:10]
            raise ValueError(
                f"{split}: {model} pair set differs from GNN. "
                f"Missing sample={missing_from_model}; extra sample={extra_in_model}"
            )

    aligned: list[dict[str, Any]] = []
    for key, gnn_row in model_rows["gnn"].items():
        sbert_row = model_rows["sbert"][key]
        keyword_row = model_rows["keyword"][key]
        actual = {
            target: actual_value("gnn", gnn_row, target) for target in TARGETS
        }
        for model, row in (("sbert", sbert_row), ("keyword", keyword_row)):
            for target in TARGETS:
                if actual_value(model, row, target) != actual[target]:
                    raise ValueError(f"{split}: target mismatch for {model}, {key}, {target}")
        aligned.append(
            {
                "candidate_id": key[0],
                "job_id": key[1],
                "actual": actual,
                "predictions": {
                    model: {
                        target: prediction_value(model, row, target) for target in TARGETS
                    }
                    for model, row in (
                        ("gnn", gnn_row),
                        ("sbert", sbert_row),
                        ("keyword", keyword_row),
                    )
                },
                "keyword_status": keyword_row.get("status"),
            }
        )
    return aligned


def arrays(
    rows: list[dict[str, Any]], target: str
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    actual = np.asarray([row["actual"][target] for row in rows], dtype=np.float64)
    candidate_ids = np.asarray([row["candidate_id"] for row in rows], dtype=object)
    predictions: dict[str, np.ndarray] = {}
    for model in ("gnn", "sbert", "keyword"):
        predictions[model] = np.asarray(
            [
                np.nan if row["predictions"][model][target] is None else row["predictions"][model][target]
                for row in rows
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(actual)):
        raise ValueError(f"Non-finite {target} target detected")
    if not np.all(np.isfinite(predictions["gnn"])):
        raise ValueError(f"Non-finite GNN {target} prediction detected")
    if not np.all(np.isfinite(predictions["sbert"])):
        raise ValueError(f"Non-finite SBERT {target} prediction detected")
    return actual, predictions, candidate_ids


def blend_predictions(
    predictions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    order = ("gnn", "sbert", "keyword")
    values = np.column_stack(
        [np.nan_to_num(predictions[model], nan=0.0) for model in order]
    )
    availability = np.column_stack(
        [np.isfinite(predictions[model]).astype(np.float64) for model in order]
    )
    weight_vector = np.asarray([weights[model] for model in order], dtype=np.float64)
    effective_weights = availability * weight_vector
    denominator = effective_weights.sum(axis=1)
    result = np.empty(len(values), dtype=np.float64)
    usable = denominator > 0
    result[usable] = (
        (values[usable] * effective_weights[usable]).sum(axis=1) / denominator[usable]
    )
    # This can occur only for a keyword-only blend on a missing-keyword row.
    result[~usable] = predictions["gnn"][~usable]
    return result


def find_best_validation_blend(
    rows: list[dict[str, Any]], target: str
) -> dict[str, Any]:
    actual, predictions, _ = arrays(rows, target)
    best_mae = float("inf")
    best_weights: dict[str, float] | None = None
    steps = int(round(1.0 / WEIGHT_STEP))
    for gnn_units in range(steps + 1):
        for sbert_units in range(steps - gnn_units + 1):
            keyword_units = steps - gnn_units - sbert_units
            weights = {
                "gnn": gnn_units / steps,
                "sbert": sbert_units / steps,
                "keyword": keyword_units / steps,
            }
            prediction = blend_predictions(predictions, weights)
            mae = float(np.mean(np.abs(prediction - actual)))
            if mae < best_mae - 1e-12:
                best_mae = mae
                best_weights = weights
    if best_weights is None:
        raise RuntimeError("Validation blend search failed")

    gnn_mae = float(np.mean(np.abs(predictions["gnn"] - actual)))
    relative_improvement = (gnn_mae - best_mae) / gnn_mae
    accepted = relative_improvement >= MINIMUM_RELATIVE_VALIDATION_IMPROVEMENT
    return {
        "weights": best_weights,
        "validation_mae": best_mae,
        "gnn_validation_mae": gnn_mae,
        "relative_validation_improvement": relative_improvement,
        "accepted": accepted,
        "selected_model": "validation_optimised_convex_blend" if accepted else "gnn",
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0 + 1.0
        position = end
    return ranks


def correlation(actual: np.ndarray, predicted: np.ndarray, spearman: bool) -> float | None:
    if spearman:
        actual = rankdata(actual)
        predicted = rankdata(predicted)
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return None
    return round(float(np.corrcoef(actual, predicted)[0, 1]), 4)


def grouped_indices(candidate_ids: np.ndarray) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, candidate_id in enumerate(candidate_ids.tolist()):
        groups[str(candidate_id)].append(index)
    return groups


def centred_ndcg(
    candidate_ids: np.ndarray, actual: np.ndarray, predicted: np.ndarray, k: int
) -> dict[str, Any]:
    scores: list[float] = []
    for indices in grouped_indices(candidate_ids).values():
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
        ideal_dcg = float(np.sum(relevance[ideal_order] * discounts))
        scores.append(dcg / ideal_dcg)
    return {
        "candidate_count": len(scores),
        "mean": round(float(np.mean(scores)), 4) if scores else None,
        "relevance": "candidate-centred linear gain",
    }


def pairwise_accuracy(
    candidate_ids: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    correct = 0.0
    comparisons = 0
    candidate_count = 0
    for indices in grouped_indices(candidate_ids).values():
        local = 0
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                true_difference = actual[left] - actual[right]
                if true_difference == 0:
                    continue
                predicted_difference = predicted[left] - predicted[right]
                comparisons += 1
                local += 1
                if predicted_difference == 0:
                    correct += 0.5
                elif np.sign(predicted_difference) == np.sign(true_difference):
                    correct += 1.0
        if local:
            candidate_count += 1
    return {
        "candidate_count": candidate_count,
        "pair_comparisons": comparisons,
        "accuracy": round(correct / comparisons, 4) if comparisons else None,
        "tie_handling": "Prediction ties receive 0.5 credit; target ties are excluded",
    }


def metrics(
    candidate_ids: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    available = np.isfinite(predicted)
    candidate_ids = candidate_ids[available]
    actual = actual[available]
    predicted = predicted[available]
    error = predicted - actual
    return {
        "evaluated_pairs": int(len(actual)),
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "pearson": correlation(actual, predicted, False),
        "spearman": correlation(actual, predicted, True),
        "mean_error": round(float(np.mean(error)), 4),
        "candidate_centred_ndcg_at_10": centred_ndcg(
            candidate_ids, actual, predicted, 10
        ),
        "pairwise_ranking_accuracy": pairwise_accuracy(
            candidate_ids, actual, predicted
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model/blend selection is completed from validation before test or gold is loaded.
    validation_rows = load_aligned_split("validation")
    selection = {
        target: find_best_validation_blend(validation_rows, target) for target in TARGETS
    }

    all_rows = {"validation": validation_rows}
    for split in ("train", "test", "gold"):
        all_rows[split] = load_aligned_split(split)

    split_reports: dict[str, Any] = {}
    for split in SPLITS:
        rows = all_rows[split]
        split_report: dict[str, Any] = {
            "pair_count": len(rows),
            "unique_candidate_count": len({row["candidate_id"] for row in rows}),
            "keyword_unavailable_pairs": sum(
                row["predictions"]["keyword"]["skills"] is None for row in rows
            ),
            "targets": {},
        }
        final_rows = [
            {
                "candidate_id": row["candidate_id"],
                "job_id": row["job_id"],
            }
            for row in rows
        ]

        for target in TARGETS:
            actual, predictions, candidate_ids = arrays(rows, target)
            candidate_blend = blend_predictions(predictions, selection[target]["weights"])
            final_prediction = (
                candidate_blend
                if selection[target]["accepted"]
                else predictions["gnn"]
            )
            split_report["targets"][target] = {
                "training_mean_dummy": None,
                "keyword": metrics(candidate_ids, actual, predictions["keyword"]),
                "sbert": metrics(candidate_ids, actual, predictions["sbert"]),
                "gnn": metrics(candidate_ids, actual, predictions["gnn"]),
                "candidate_blend": metrics(candidate_ids, actual, candidate_blend),
                "selected_final": metrics(candidate_ids, actual, final_prediction),
                "selected_model": selection[target]["selected_model"],
            }
            for index, row in enumerate(final_rows):
                row[f"{target}_actual"] = round(float(actual[index]), 6)
                row[f"{target}_gnn"] = round(float(predictions["gnn"][index]), 6)
                row[f"{target}_sbert"] = round(float(predictions["sbert"][index]), 6)
                row[f"{target}_keyword"] = (
                    None
                    if not np.isfinite(predictions["keyword"][index])
                    else round(float(predictions["keyword"][index]), 6)
                )
                row[f"{target}_candidate_blend"] = round(
                    float(candidate_blend[index]), 6
                )
                row[f"{target}_final"] = round(float(final_prediction[index]), 6)
        split_reports[split] = split_report
        save_json(OUTPUT_DIR / f"{split}_predictions.json", final_rows)

    # Add one fixed dummy prediction learned only from training targets.
    training_means = {
        target: float(
            np.mean([row["actual"][target] for row in all_rows["train"]])
        )
        for target in TARGETS
    }
    for split in SPLITS:
        rows = all_rows[split]
        for target in TARGETS:
            actual, _, candidate_ids = arrays(rows, target)
            dummy = np.full(len(actual), training_means[target], dtype=np.float64)
            split_reports[split]["targets"][target]["training_mean_dummy"] = metrics(
                candidate_ids, actual, dummy
            )

    report = {
        "stage": "Final model selection and baseline comparison",
        "decision": {
            target: {
                "selected_model": values["selected_model"],
                "candidate_blend_weights": values["weights"],
                "gnn_validation_mae": round(values["gnn_validation_mae"], 6),
                "candidate_blend_validation_mae": round(values["validation_mae"], 6),
                "relative_validation_improvement": round(
                    values["relative_validation_improvement"], 6
                ),
                "minimum_required_improvement": MINIMUM_RELATIVE_VALIDATION_IMPROVEMENT,
                "candidate_blend_accepted": values["accepted"],
            }
            for target, values in selection.items()
        },
        "final_architecture": {
            "numeric_scoring": "SBERT-powered weighted heterogeneous GNN",
            "numeric_checkpoint": "models/gnn/heterogeneous_gnn.pt",
            "explanations": "Deterministic keyword matched and missing skill evidence",
            "keyword_score_used_in_final_numeric_score": any(
                values["accepted"] and values["weights"]["keyword"] > 0
                for values in selection.values()
            ),
            "forced_ensemble": False,
        },
        "selection_protocol": {
            "weight_search": "Convex weights on a 0.01 grid",
            "objective": "Validation MAE, independently for each target",
            "complexity_guardrail": "Accept a blend only with at least 1% relative validation-MAE improvement over GNN",
            "missing_keyword_handling": "Renormalise available component weights; GNN fallback if none remain",
        },
        "training_mean_dummy_values": {
            target: round(value, 6) for target, value in training_means.items()
        },
        "splits": split_reports,
        "integrity_checks": {
            "pair_sets_identical_across_models": True,
            "duplicate_pairs": 0,
            "target_mismatches": 0,
            "sbert_missing_predictions": 0,
            "gnn_missing_predictions": 0,
            "keyword_missing_predictions_expected_from_unavailable_jobs": True,
        },
        "leakage_controls": {
            "blend_weights_selected_from_validation_only": True,
            "test_used_for_selection": False,
            "gold_used_for_selection": False,
            "gold_used_for_weight_tuning": False,
            "test_and_gold_loaded_after_selection_was_frozen": True,
        },
        "limitations": [
            "Role Radar training labels are generated compatibility labels, not observed hiring outcomes.",
            "The human-reviewed gold set contains only 108 pairs and limited within-candidate ranking comparisons.",
            "Gold targets have a higher mean than generated train, validation and test targets.",
        ],
    }
    save_json(OUTPUT_DIR / "final_model_report.json", report)

    config = {
        "model_name": "Job Optimizer Semantic Graph Matcher",
        "version": "1.0.0",
        "numeric_model": {
            "type": "sbert_weighted_heterogeneous_gnn",
            "checkpoint": "models/gnn/heterogeneous_gnn.pt",
            "target_selection": {
                target: selection[target]["selected_model"] for target in TARGETS
            },
        },
        "explanation_model": {
            "type": "deterministic_weighted_skill_evidence",
            "candidate_weights": {"primary": 1.0, "secondary": 0.75},
            "job_weights": {"required": 1.0, "preferred": 0.6, "mentioned": 0.25},
            "purpose": "Display matched skills and missing required/preferred skills; do not fabricate evidence.",
        },
        "score_policy": {
            "artificial_fixed_offset": False,
            "artificial_score_cap_during_evaluation": False,
            "gold_used_for_tuning": False,
        },
        "evaluation_report": "evaluation/final_model/final_model_report.json",
    }
    save_json(MODEL_DIR / "model_config.json", config)

    print("Final model selection completed.")
    for target in TARGETS:
        values = report["decision"][target]
        print(
            f"{target.title()}: {values['selected_model']} | "
            f"candidate blend improvement={values['relative_validation_improvement'] * 100:.3f}%"
        )
    print(f"Report: {OUTPUT_DIR / 'final_model_report.json'}")
    print(f"Configuration: {MODEL_DIR / 'model_config.json'}")


if __name__ == "__main__":
    main()
