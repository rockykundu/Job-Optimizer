import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "keyword_baseline"
)


def load_json(filename):
    with (BASELINE_DIR / filename).open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json(filename, data):
    with (BASELINE_DIR / filename).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def fit_affine_calibrator(
    predictions,
    target_field,
):
    usable = [
        item
        for item in predictions
        if item.get("keyword_skill_score")
        is not None
        and item.get(target_field) is not None
    ]

    x_values = [
        item["keyword_skill_score"]
        for item in usable
    ]

    y_values = [
        item[target_field]
        for item in usable
    ]

    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    variance_x = sum(
        (value - mean_x) ** 2
        for value in x_values
    )

    covariance = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(x_values, y_values)
    )

    if variance_x == 0:
        slope = 0.0
    else:
        slope = covariance / variance_x

    intercept = mean_y - slope * mean_x

    return {
        "slope": slope,
        "intercept": intercept,
        "fitted_pairs": len(usable),
        "training_input_mean": mean_x,
        "training_target_mean": mean_y,
    }


def apply_calibrator(raw_score, calibrator):
    if raw_score is None:
        return None

    score = (
        calibrator["slope"] * raw_score
        + calibrator["intercept"]
    )

    return round(
        min(100.0, max(0.0, score)),
        4,
    )


def pearson(x_values, y_values):
    if len(x_values) < 2:
        return None

    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(x_values, y_values)
    )

    denominator_x = math.sqrt(
        sum(
            (x - mean_x) ** 2
            for x in x_values
        )
    )

    denominator_y = math.sqrt(
        sum(
            (y - mean_y) ** 2
            for y in y_values
        )
    )

    denominator = denominator_x * denominator_y

    if denominator == 0:
        return None

    return numerator / denominator


def rank_values(values):
    sorted_values = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(values)
    position = 0

    while position < len(sorted_values):
        end = position

        while (
            end + 1 < len(sorted_values)
            and sorted_values[end + 1][1]
            == sorted_values[position][1]
        ):
            end += 1

        average_rank = (
            position + end + 2
        ) / 2.0

        for index in range(position, end + 1):
            original_index = sorted_values[index][0]
            ranks[original_index] = average_rank

        position = end + 1

    return ranks


def regression_metrics(
    predictions,
    prediction_field,
    target_field,
):
    usable = [
        item
        for item in predictions
        if item.get(prediction_field) is not None
        and item.get(target_field) is not None
    ]

    if not usable:
        return {
            "evaluated_pairs": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
        }

    predicted = [
        item[prediction_field]
        for item in usable
    ]

    targets = [
        item[target_field]
        for item in usable
    ]

    absolute_errors = [
        abs(prediction - target)
        for prediction, target
        in zip(predicted, targets)
    ]

    squared_errors = [
        (prediction - target) ** 2
        for prediction, target
        in zip(predicted, targets)
    ]

    pearson_value = pearson(
        predicted,
        targets,
    )

    spearman_value = pearson(
        rank_values(predicted),
        rank_values(targets),
    )

    return {
        "evaluated_pairs": len(usable),
        "mae": round(
            statistics.mean(absolute_errors),
            4,
        ),
        "rmse": round(
            math.sqrt(
                statistics.mean(squared_errors)
            ),
            4,
        ),
        "pearson": (
            round(pearson_value, 4)
            if pearson_value is not None
            else None
        ),
        "spearman": (
            round(spearman_value, 4)
            if spearman_value is not None
            else None
        ),
    }


def score_distribution(
    predictions,
    field,
):
    values = [
        item[field]
        for item in predictions
        if item.get(field) is not None
    ]

    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(values),
        "minimum": round(min(values), 4),
        "maximum": round(max(values), 4),
        "mean": round(
            statistics.mean(values),
            4,
        ),
        "median": round(
            statistics.median(values),
            4,
        ),
    }


def dcg(relevances, cutoff):
    return sum(
        relevance / math.log2(index + 2)
        for index, relevance
        in enumerate(relevances[:cutoff])
    )


def candidate_centred_ndcg(
    predictions,
    prediction_field,
    target_field,
    cutoff,
):
    by_candidate = defaultdict(list)

    for item in predictions:
        if (
            item.get(prediction_field) is not None
            and item.get(target_field) is not None
        ):
            by_candidate[
                item["profile_id"]
            ].append(item)

    ndcg_values = []

    for candidate_items in by_candidate.values():
        if len(candidate_items) < 2:
            continue

        minimum_target = min(
            item[target_field]
            for item in candidate_items
        )

        # The least relevant job receives zero gain.
        def relevance(item):
            return max(
                0.0,
                item[target_field]
                - minimum_target,
            )

        ideal_order = sorted(
            candidate_items,
            key=lambda item: item[target_field],
            reverse=True,
        )

        predicted_order = sorted(
            candidate_items,
            key=lambda item:
                item[prediction_field],
            reverse=True,
        )

        ideal_relevances = [
            relevance(item)
            for item in ideal_order
        ]

        predicted_relevances = [
            relevance(item)
            for item in predicted_order
        ]

        ideal_dcg = dcg(
            ideal_relevances,
            cutoff,
        )

        if ideal_dcg == 0:
            continue

        predicted_dcg = dcg(
            predicted_relevances,
            cutoff,
        )

        ndcg_values.append(
            predicted_dcg / ideal_dcg
        )

    return {
        "candidate_count": len(ndcg_values),
        "mean": (
            round(
                statistics.mean(ndcg_values),
                4,
            )
            if ndcg_values
            else None
        ),
    }


def pairwise_ranking_accuracy(
    predictions,
    prediction_field,
    target_field,
):
    by_candidate = defaultdict(list)

    for item in predictions:
        if (
            item.get(prediction_field) is not None
            and item.get(target_field) is not None
        ):
            by_candidate[
                item["profile_id"]
            ].append(item)

    comparison_count = 0
    correct_score = 0.0
    evaluated_candidates = 0

    for candidate_items in by_candidate.values():
        candidate_comparisons = 0

        for first_index in range(
            len(candidate_items)
        ):
            for second_index in range(
                first_index + 1,
                len(candidate_items),
            ):
                first = candidate_items[first_index]
                second = candidate_items[second_index]

                target_difference = (
                    first[target_field]
                    - second[target_field]
                )

                if target_difference == 0:
                    continue

                prediction_difference = (
                    first[prediction_field]
                    - second[prediction_field]
                )

                comparison_count += 1
                candidate_comparisons += 1

                if prediction_difference == 0:
                    correct_score += 0.5
                elif (
                    prediction_difference
                    * target_difference
                    > 0
                ):
                    correct_score += 1.0

        if candidate_comparisons:
            evaluated_candidates += 1

    return {
        "candidate_count": evaluated_candidates,
        "pair_comparisons": comparison_count,
        "accuracy": (
            round(
                correct_score / comparison_count,
                4,
            )
            if comparison_count
            else None
        ),
        "tie_handling":
            "Prediction ties receive 0.5 credit",
    }


def evaluate_target(
    predictions,
    raw_prediction_field,
    calibrated_prediction_field,
    target_field,
):
    return {
        "raw_regression": regression_metrics(
            predictions,
            raw_prediction_field,
            target_field,
        ),
        "calibrated_regression":
            regression_metrics(
                predictions,
                calibrated_prediction_field,
                target_field,
            ),
        "candidate_centred_ndcg": {
            "ndcg_at_5":
                candidate_centred_ndcg(
                    predictions,
                    raw_prediction_field,
                    target_field,
                    5,
                ),
            "ndcg_at_10":
                candidate_centred_ndcg(
                    predictions,
                    raw_prediction_field,
                    target_field,
                    10,
                ),
        },
        "pairwise_ranking_accuracy":
            pairwise_ranking_accuracy(
                predictions,
                raw_prediction_field,
                target_field,
            ),
    }


def main():
    split_names = [
        "train",
        "validation",
        "test",
        "gold",
    ]

    predictions_by_split = {
        split_name: load_json(
            f"{split_name}_predictions.json"
        )
        for split_name in split_names
    }

    training_predictions = (
        predictions_by_split["train"]
    )

    skills_calibrator = fit_affine_calibrator(
        training_predictions,
        "skills_score",
    )

    composite_calibrator = fit_affine_calibrator(
        training_predictions,
        "composite_score",
    )

    split_reports = {}

    for split_name, predictions in (
        predictions_by_split.items()
    ):
        calibrated_predictions = []

        for original in predictions:
            item = dict(original)

            item[
                "calibrated_skills_prediction"
            ] = apply_calibrator(
                item.get("keyword_skill_score"),
                skills_calibrator,
            )

            item[
                "calibrated_composite_prediction"
            ] = apply_calibrator(
                item.get("keyword_skill_score"),
                composite_calibrator,
            )

            calibrated_predictions.append(item)

        save_json(
            f"{split_name}_predictions_calibrated.json",
            calibrated_predictions,
        )

        split_reports[split_name] = {
            "available_pairs": sum(
                item.get("keyword_skill_score")
                is not None
                for item in calibrated_predictions
            ),
            "raw_score_distribution":
                score_distribution(
                    calibrated_predictions,
                    "keyword_skill_score",
                ),
            "calibrated_skills_distribution":
                score_distribution(
                    calibrated_predictions,
                    "calibrated_skills_prediction",
                ),
            "skills_target": evaluate_target(
                calibrated_predictions,
                "keyword_skill_score",
                "calibrated_skills_prediction",
                "skills_score",
            ),
            "composite_target": evaluate_target(
                calibrated_predictions,
                "keyword_skill_score",
                "calibrated_composite_prediction",
                "composite_score",
            ),
        }

    report = {
        "model": {
            "name":
                "Calibrated keyword baseline",
            "raw_model":
                "Deterministic weighted skill coverage",
            "calibration":
                "Affine calibration fitted on training only",
        },
        "calibrators": {
            "skills_target":
                skills_calibrator,
            "composite_target":
                composite_calibrator,
        },
        "splits": split_reports,
        "ranking_correction": {
            "old_ndcg_status":
                "Discarded because raw relevance values "
                "between 50 and 100 inflated NDCG",
            "new_ndcg":
                "Candidate-centred gain where each "
                "candidate's minimum label receives zero gain",
            "additional_metric":
                "Pairwise ranking accuracy",
        },
        "safety_rules": {
            "calibration_fitted_on_training_only":
                True,
            "validation_used_for_calibration":
                False,
            "test_used_for_calibration":
                False,
            "gold_used_for_calibration":
                False,
            "artificial_fixed_offset":
                False,
            "artificial_score_cap":
                False,
        },
    }

    save_json(
        "keyword_baseline_corrected_report.json",
        report,
    )

    print(
        "Corrected keyword baseline evaluation completed."
    )
    print("")

    print(
        "Skills calibration formula:"
    )

    print(
        "calibrated = "
        f"{skills_calibrator['slope']:.6f} "
        "* raw_score + "
        f"{skills_calibrator['intercept']:.6f}"
    )

    print("")

    for split_name, split_report in (
        split_reports.items()
    ):
        metrics = (
            split_report["skills_target"]
        )

        print(split_name.title())

        print(
            "  Calibrated skills MAE: "
            f"{metrics['calibrated_regression']['mae']}"
        )

        print(
            "  Centred NDCG@10: "
            f"{metrics['candidate_centred_ndcg']['ndcg_at_10']['mean']}"
        )

        print(
            "  Pairwise accuracy: "
            f"{metrics['pairwise_ranking_accuracy']['accuracy']}"
        )

    print("")
    print(
        "Report created at: "
        f"{BASELINE_DIR / 'keyword_baseline_corrected_report.json'}"
    )


if __name__ == "__main__":
    main()