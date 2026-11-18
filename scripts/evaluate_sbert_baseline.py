import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "sbert"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "sbert"


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


def fit_affine_calibrator(
    predictions,
    input_field,
    target_field,
):
    usable = [
        item
        for item in predictions
        if item.get(input_field) is not None
        and item.get(target_field) is not None
    ]

    x_values = [
        item[input_field]
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

    slope = (
        covariance / variance_x
        if variance_x
        else 0.0
    )

    intercept = mean_y - slope * mean_x

    return {
        "slope": slope,
        "intercept": intercept,
        "fitted_pairs": len(usable),
        "training_input_mean": mean_x,
        "training_target_mean": mean_y,
    }


def apply_calibrator(value, calibrator):
    if value is None:
        return None

    calibrated = (
        calibrator["slope"] * value
        + calibrator["intercept"]
    )

    return round(
        min(100.0, max(0.0, calibrated)),
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
    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(values)
    position = 0

    while position < len(indexed):
        end = position

        while (
            end + 1 < len(indexed)
            and indexed[end + 1][1]
            == indexed[position][1]
        ):
            end += 1

        average_rank = (
            position + end + 2
        ) / 2.0

        for index in range(position, end + 1):
            original_index = indexed[index][0]
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

    target = [
        item[target_field]
        for item in usable
    ]

    absolute_errors = [
        abs(x - y)
        for x, y in zip(predicted, target)
    ]

    squared_errors = [
        (x - y) ** 2
        for x, y in zip(predicted, target)
    ]

    pearson_value = pearson(
        predicted,
        target,
    )

    spearman_value = pearson(
        rank_values(predicted),
        rank_values(target),
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
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
        "mean": round(
            statistics.mean(values),
            6,
        ),
        "median": round(
            statistics.median(values),
            6,
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
    grouped = defaultdict(list)

    for item in predictions:
        if (
            item.get(prediction_field) is not None
            and item.get(target_field) is not None
        ):
            grouped[
                item["profile_id"]
            ].append(item)

    scores = []

    for candidate_items in grouped.values():
        if len(candidate_items) < 2:
            continue

        minimum_target = min(
            item[target_field]
            for item in candidate_items
        )

        def relevance(item):
            return max(
                0.0,
                item[target_field]
                - minimum_target,
            )

        predicted_order = sorted(
            candidate_items,
            key=lambda item:
                item[prediction_field],
            reverse=True,
        )

        ideal_order = sorted(
            candidate_items,
            key=lambda item:
                item[target_field],
            reverse=True,
        )

        predicted_relevance = [
            relevance(item)
            for item in predicted_order
        ]

        ideal_relevance = [
            relevance(item)
            for item in ideal_order
        ]

        ideal_value = dcg(
            ideal_relevance,
            cutoff,
        )

        if ideal_value == 0:
            continue

        scores.append(
            dcg(
                predicted_relevance,
                cutoff,
            )
            / ideal_value
        )

    return {
        "candidate_count": len(scores),
        "mean": (
            round(statistics.mean(scores), 4)
            if scores
            else None
        ),
    }


def pairwise_ranking_accuracy(
    predictions,
    prediction_field,
    target_field,
):
    grouped = defaultdict(list)

    for item in predictions:
        if (
            item.get(prediction_field) is not None
            and item.get(target_field) is not None
        ):
            grouped[
                item["profile_id"]
            ].append(item)

    comparisons = 0
    correct_score = 0.0
    evaluated_candidates = 0

    for candidate_items in grouped.values():
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

                comparisons += 1
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
        "candidate_count":
            evaluated_candidates,
        "pair_comparisons":
            comparisons,
        "accuracy": (
            round(
                correct_score / comparisons,
                4,
            )
            if comparisons
            else None
        ),
        "tie_handling":
            "Prediction ties receive 0.5 credit",
    }


def constant_baseline_metrics(
    predictions,
    target_field,
    training_mean,
):
    temporary_field = (
        f"_constant_{target_field}"
    )

    for item in predictions:
        item[temporary_field] = training_mean

    metrics = regression_metrics(
        predictions,
        temporary_field,
        target_field,
    )

    for item in predictions:
        item.pop(temporary_field, None)

    return metrics


def create_raw_predictions(
    labels,
    candidate_embeddings,
    job_embeddings,
    candidate_index,
    job_index,
):
    predictions = []
    unavailable = []

    for label in labels:
        profile_id = label["profile_id"]
        job_id = label["job_id"]

        candidate_position = (
            candidate_index.get(profile_id)
        )

        job_position = job_index.get(job_id)

        if (
            candidate_position is None
            or job_position is None
        ):
            unavailable.append(
                label["pair_id"]
            )
            continue

        cosine_similarity = float(
            np.dot(
                candidate_embeddings[
                    candidate_position
                ],
                job_embeddings[
                    job_position
                ],
            )
        )

        predictions.append(
            {
                "pair_id": label["pair_id"],
                "profile_id": profile_id,
                "job_id": job_id,
                "cosine_similarity":
                    cosine_similarity,
                "semantic_score_0_100":
                    round(
                        (
                            cosine_similarity + 1.0
                        )
                        * 50.0,
                        4,
                    ),
                "skills_score":
                    label["skills_score"],
                "composite_score":
                    label["composite_score"],
            }
        )

    return predictions, unavailable


def evaluate_split(
    split_name,
    predictions,
    skills_calibrator,
    composite_calibrator,
    training_skills_mean,
    training_composite_mean,
):
    for item in predictions:
        item[
            "calibrated_skills_prediction"
        ] = apply_calibrator(
            item["cosine_similarity"],
            skills_calibrator,
        )

        item[
            "calibrated_composite_prediction"
        ] = apply_calibrator(
            item["cosine_similarity"],
            composite_calibrator,
        )

    save_json(
        OUTPUT_DIR
        / f"{split_name}_predictions.json",
        predictions,
    )

    return {
        "pair_count": len(predictions),
        "cosine_distribution":
            score_distribution(
                predictions,
                "cosine_similarity",
            ),
        "skills_target": {
            "training_mean_baseline":
                constant_baseline_metrics(
                    predictions,
                    "skills_score",
                    training_skills_mean,
                ),
            "sbert_calibrated_regression":
                regression_metrics(
                    predictions,
                    "calibrated_skills_prediction",
                    "skills_score",
                ),
            "candidate_centred_ndcg": {
                "ndcg_at_5":
                    candidate_centred_ndcg(
                        predictions,
                        "cosine_similarity",
                        "skills_score",
                        5,
                    ),
                "ndcg_at_10":
                    candidate_centred_ndcg(
                        predictions,
                        "cosine_similarity",
                        "skills_score",
                        10,
                    ),
            },
            "pairwise_ranking_accuracy":
                pairwise_ranking_accuracy(
                    predictions,
                    "cosine_similarity",
                    "skills_score",
                ),
        },
        "composite_target": {
            "training_mean_baseline":
                constant_baseline_metrics(
                    predictions,
                    "composite_score",
                    training_composite_mean,
                ),
            "sbert_calibrated_regression":
                regression_metrics(
                    predictions,
                    "calibrated_composite_prediction",
                    "composite_score",
                ),
            "candidate_centred_ndcg": {
                "ndcg_at_5":
                    candidate_centred_ndcg(
                        predictions,
                        "cosine_similarity",
                        "composite_score",
                        5,
                    ),
                "ndcg_at_10":
                    candidate_centred_ndcg(
                        predictions,
                        "cosine_similarity",
                        "composite_score",
                        10,
                    ),
            },
            "pairwise_ranking_accuracy":
                pairwise_ranking_accuracy(
                    predictions,
                    "cosine_similarity",
                    "composite_score",
                ),
        },
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_embeddings = np.load(
        MODEL_DIR
        / "candidate_embeddings.npy"
    )

    job_embeddings = np.load(
        MODEL_DIR / "job_embeddings.npy"
    )

    candidate_ids = load_json(
        MODEL_DIR / "candidate_ids.json"
    )

    job_ids = load_json(
        MODEL_DIR / "job_ids.json"
    )

    candidate_index = {
        int(candidate_id): index
        for index, candidate_id
        in enumerate(candidate_ids)
    }

    job_index = {
        job_id: index
        for index, job_id
        in enumerate(job_ids)
    }

    split_paths = {
        "train":
            SPLIT_DIR / "train_labels.json",
        "validation":
            SPLIT_DIR / "validation_labels.json",
        "test":
            SPLIT_DIR / "test_labels.json",
        "gold":
            SPLIT_DIR / "gold_evaluation.json",
    }

    raw_predictions = {}
    unavailable_by_split = {}

    for split_name, split_path in (
        split_paths.items()
    ):
        labels = load_json(split_path)

        (
            raw_predictions[split_name],
            unavailable_by_split[split_name],
        ) = create_raw_predictions(
            labels,
            candidate_embeddings,
            job_embeddings,
            candidate_index,
            job_index,
        )

    training_predictions = (
        raw_predictions["train"]
    )

    skills_calibrator = (
        fit_affine_calibrator(
            training_predictions,
            "cosine_similarity",
            "skills_score",
        )
    )

    composite_calibrator = (
        fit_affine_calibrator(
            training_predictions,
            "cosine_similarity",
            "composite_score",
        )
    )

    training_skills_mean = (
        skills_calibrator[
            "training_target_mean"
        ]
    )

    training_composite_mean = (
        composite_calibrator[
            "training_target_mean"
        ]
    )

    split_reports = {}

    for split_name, predictions in (
        raw_predictions.items()
    ):
        split_reports[split_name] = (
            evaluate_split(
                split_name,
                predictions,
                skills_calibrator,
                composite_calibrator,
                training_skills_mean,
                training_composite_mean,
            )
        )

        split_reports[split_name][
            "unavailable_pair_count"
        ] = len(
            unavailable_by_split[split_name]
        )

        split_reports[split_name][
            "unavailable_pair_ids"
        ] = unavailable_by_split[
            split_name
        ]

    report = {
        "model": {
            "name": "all-MiniLM-L6-v2",
            "baseline":
                "Chunked structured-text SBERT",
            "similarity":
                "Cosine similarity of normalised "
                "candidate and job embeddings",
            "calibration":
                "Separate affine calibrators fitted "
                "on training labels only",
        },
        "calibrators": {
            "skills_target":
                skills_calibrator,
            "composite_target":
                composite_calibrator,
        },
        "splits": split_reports,
        "embedding_shapes": {
            "candidates":
                list(candidate_embeddings.shape),
            "jobs":
                list(job_embeddings.shape),
        },
        "safety_rules": {
            "training_labels_used_for_calibration_only":
                True,
            "validation_used_for_calibration":
                False,
            "test_used_for_calibration":
                False,
            "gold_used_for_calibration":
                False,
            "long_documents_chunked":
                True,
            "artificial_fixed_offset":
                False,
            "gold_pairs_missing_embeddings":
                len(
                    unavailable_by_split["gold"]
                ),
        },
    }

    save_json(
        OUTPUT_DIR / "sbert_report.json",
        report,
    )

    print(
        "Stage 7B SBERT evaluation completed."
    )
    print("")

    for split_name, split_report in (
        split_reports.items()
    ):
        skills = split_report[
            "skills_target"
        ]

        composite = split_report[
            "composite_target"
        ]

        print(split_name.title())

        print(
            "  Skills MAE: "
            f"{skills['sbert_calibrated_regression']['mae']}"
        )

        print(
            "  Composite MAE: "
            f"{composite['sbert_calibrated_regression']['mae']}"
        )

        print(
            "  Composite NDCG@10: "
            f"{composite['candidate_centred_ndcg']['ndcg_at_10']['mean']}"
        )

        print(
            "  Composite pairwise accuracy: "
            f"{composite['pairwise_ranking_accuracy']['accuracy']}"
        )

    print("")
    print(
        "Report created at: "
        f"{OUTPUT_DIR / 'sbert_report.json'}"
    )


if __name__ == "__main__":
    main()