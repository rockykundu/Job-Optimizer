import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "keyword_baseline"
)


CANDIDATE_PRIMARY_WEIGHT = 1.00
CANDIDATE_SECONDARY_WEIGHT = 0.75

JOB_REQUIRED_WEIGHT = 1.00
JOB_PREFERRED_WEIGHT = 0.60
JOB_MENTIONED_WEIGHT = 0.25


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


def build_candidate_skill_map(profile):
    skill_map = {}

    for skill in profile.get(
        "skills_primary",
        [],
    ):
        skill_map[skill.casefold()] = {
            "name": skill,
            "weight": CANDIDATE_PRIMARY_WEIGHT,
            "category": "primary",
        }

    for skill in profile.get(
        "skills_secondary",
        [],
    ):
        key = skill.casefold()

        if key not in skill_map:
            skill_map[key] = {
                "name": skill,
                "weight":
                    CANDIDATE_SECONDARY_WEIGHT,
                "category": "secondary",
            }

    return skill_map


def build_job_skill_map(job):
    skill_map = {}

    categories = [
        (
            "required",
            job.get("skills_required", []),
            JOB_REQUIRED_WEIGHT,
        ),
        (
            "preferred",
            job.get("skills_preferred", []),
            JOB_PREFERRED_WEIGHT,
        ),
        (
            "mentioned",
            job.get("skills_mentioned", []),
            JOB_MENTIONED_WEIGHT,
        ),
    ]

    for category, skills, weight in categories:
        for skill in skills:
            key = skill.casefold()

            existing = skill_map.get(key)

            if (
                existing is None
                or weight > existing["weight"]
            ):
                skill_map[key] = {
                    "name": skill,
                    "weight": weight,
                    "category": category,
                }

    return skill_map


def predict_keyword_score(profile, job):
    candidate_skills = (
        build_candidate_skill_map(profile)
    )

    job_skills = build_job_skill_map(job)

    if not job_skills:
        return {
            "status": "job_has_no_detected_skills",
            "keyword_skill_score": None,
            "matched_skills": [],
            "missing_required_skills": [],
            "missing_preferred_skills": [],
            "missing_mentioned_skills": [],
        }

    denominator = sum(
        item["weight"]
        for item in job_skills.values()
    )

    numerator = 0.0
    matched_skills = []
    missing_required = []
    missing_preferred = []
    missing_mentioned = []

    for skill_key, job_skill in job_skills.items():
        candidate_skill = candidate_skills.get(
            skill_key
        )

        if candidate_skill:
            contribution = (
                job_skill["weight"]
                * candidate_skill["weight"]
            )

            numerator += contribution

            matched_skills.append(
                {
                    "skill": job_skill["name"],
                    "job_category":
                        job_skill["category"],
                    "candidate_category":
                        candidate_skill["category"],
                    "contribution":
                        round(contribution, 4),
                }
            )
        else:
            if job_skill["category"] == "required":
                missing_required.append(
                    job_skill["name"]
                )
            elif (
                job_skill["category"]
                == "preferred"
            ):
                missing_preferred.append(
                    job_skill["name"]
                )
            else:
                missing_mentioned.append(
                    job_skill["name"]
                )

    score = (
        numerator / denominator * 100
        if denominator > 0
        else None
    )

    return {
        "status": "success",
        "keyword_skill_score":
            round(score, 4),
        "matched_skills": sorted(
            matched_skills,
            key=lambda item:
                item["skill"].casefold(),
        ),
        "missing_required_skills": sorted(
            missing_required,
            key=str.casefold,
        ),
        "missing_preferred_skills": sorted(
            missing_preferred,
            key=str.casefold,
        ),
        "missing_mentioned_skills": sorted(
            missing_mentioned,
            key=str.casefold,
        ),
    }


def pearson_correlation(x_values, y_values):
    if len(x_values) < 2:
        return None

    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(
            x_values,
            y_values,
        )
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

    denominator = (
        denominator_x * denominator_y
    )

    if denominator == 0:
        return None

    return numerator / denominator


def rank_values(values):
    indexed_values = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(values)
    position = 0

    while position < len(indexed_values):
        end = position

        while (
            end + 1 < len(indexed_values)
            and indexed_values[end + 1][1]
            == indexed_values[position][1]
        ):
            end += 1

        average_rank = (
            position + end + 2
        ) / 2.0

        for index in range(position, end + 1):
            original_index = (
                indexed_values[index][0]
            )

            ranks[original_index] = (
                average_rank
            )

        position = end + 1

    return ranks


def spearman_correlation(
    x_values,
    y_values,
):
    if len(x_values) < 2:
        return None

    x_ranks = rank_values(x_values)
    y_ranks = rank_values(y_values)

    return pearson_correlation(
        x_ranks,
        y_ranks,
    )


def regression_metrics(
    predictions,
    target_field,
):
    available = [
        prediction
        for prediction in predictions
        if prediction["keyword_skill_score"]
        is not None
        and prediction.get(target_field)
        is not None
    ]

    if not available:
        return {
            "evaluated_pairs": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
        }

    predicted_values = [
        item["keyword_skill_score"]
        for item in available
    ]

    target_values = [
        item[target_field]
        for item in available
    ]

    absolute_errors = [
        abs(predicted - target)
        for predicted, target in zip(
            predicted_values,
            target_values,
        )
    ]

    squared_errors = [
        (predicted - target) ** 2
        for predicted, target in zip(
            predicted_values,
            target_values,
        )
    ]

    pearson = pearson_correlation(
        predicted_values,
        target_values,
    )

    spearman = spearman_correlation(
        predicted_values,
        target_values,
    )

    return {
        "evaluated_pairs":
            len(available),
        "mae": round(
            statistics.mean(
                absolute_errors
            ),
            4,
        ),
        "rmse": round(
            math.sqrt(
                statistics.mean(
                    squared_errors
                )
            ),
            4,
        ),
        "pearson": (
            round(pearson, 4)
            if pearson is not None
            else None
        ),
        "spearman": (
            round(spearman, 4)
            if spearman is not None
            else None
        ),
    }


def dcg(relevances, cutoff):
    selected = relevances[:cutoff]

    return sum(
        relevance
        / math.log2(index + 2)
        for index, relevance
        in enumerate(selected)
    )


def ndcg_metrics(
    predictions,
    target_field,
    cutoffs=(5, 10),
):
    predictions_by_candidate = (
        defaultdict(list)
    )

    for prediction in predictions:
        if (
            prediction["keyword_skill_score"]
            is None
            or prediction.get(target_field)
            is None
        ):
            continue

        predictions_by_candidate[
            prediction["profile_id"]
        ].append(prediction)

    results = {}

    for cutoff in cutoffs:
        candidate_scores = []

        for candidate_predictions in (
            predictions_by_candidate.values()
        ):
            if len(candidate_predictions) < 2:
                continue

            predicted_order = sorted(
                candidate_predictions,
                key=lambda item:
                    item["keyword_skill_score"],
                reverse=True,
            )

            ideal_order = sorted(
                candidate_predictions,
                key=lambda item:
                    item[target_field],
                reverse=True,
            )

            predicted_relevances = [
                item[target_field] / 100.0
                for item in predicted_order
            ]

            ideal_relevances = [
                item[target_field] / 100.0
                for item in ideal_order
            ]

            predicted_dcg = dcg(
                predicted_relevances,
                cutoff,
            )

            ideal_dcg = dcg(
                ideal_relevances,
                cutoff,
            )

            if ideal_dcg > 0:
                candidate_scores.append(
                    predicted_dcg / ideal_dcg
                )

        results[f"ndcg_at_{cutoff}"] = {
            "candidate_count":
                len(candidate_scores),
            "mean": (
                round(
                    statistics.mean(
                        candidate_scores
                    ),
                    4,
                )
                if candidate_scores
                else None
            ),
        }

    return results


def evaluate_split(
    split_name,
    labels,
    profiles_by_id,
    jobs_by_id,
):
    predictions = []
    unavailable_reasons = Counter()

    for label in labels:
        profile = profiles_by_id.get(
            label["profile_id"]
        )

        job = jobs_by_id.get(
            label["job_id"]
        )

        if profile is None:
            unavailable_reasons[
                "profile_not_available"
            ] += 1
            continue

        if job is None:
            unavailable_reasons[
                "job_not_available"
            ] += 1

            predictions.append(
                {
                    "pair_id":
                        label["pair_id"],
                    "profile_id":
                        label["profile_id"],
                    "job_id":
                        label["job_id"],
                    "status":
                        "job_not_available",
                    "keyword_skill_score":
                        None,
                    "skills_score":
                        label["skills_score"],
                    "composite_score":
                        label["composite_score"],
                    "matched_skills": [],
                    "missing_required_skills":
                        [],
                    "missing_preferred_skills":
                        [],
                    "missing_mentioned_skills":
                        [],
                }
            )

            continue

        prediction = predict_keyword_score(
            profile,
            job,
        )

        if prediction["status"] != "success":
            unavailable_reasons[
                prediction["status"]
            ] += 1

        prediction.update(
            {
                "pair_id": label["pair_id"],
                "profile_id":
                    label["profile_id"],
                "job_id": label["job_id"],
                "job_title":
                    job.get("title"),
                "job_role_family":
                    job.get(
                        "derived_role_family"
                    ),
                "skills_score":
                    label["skills_score"],
                "composite_score":
                    label["composite_score"],
            }
        )

        predictions.append(prediction)

    report = {
        "split": split_name,
        "input_labels": len(labels),
        "prediction_records":
            len(predictions),
        "available_scores": sum(
            prediction[
                "keyword_skill_score"
            ]
            is not None
            for prediction in predictions
        ),
        "unavailable_reasons":
            dict(unavailable_reasons),
        "skills_target_metrics":
            regression_metrics(
                predictions,
                "skills_score",
            ),
        "composite_target_metrics":
            regression_metrics(
                predictions,
                "composite_score",
            ),
        "skills_target_ranking":
            ndcg_metrics(
                predictions,
                "skills_score",
            ),
        "composite_target_ranking":
            ndcg_metrics(
                predictions,
                "composite_score",
            ),
    }

    save_json(
        OUTPUT_DIR
        / f"{split_name}_predictions.json",
        predictions,
    )

    return report


def main():
    profiles = load_json(
        PROCESSED_DIR
        / "profiles_features_enriched.json"
    )

    jobs = load_json(
        PROCESSED_DIR
        / "jobs_features_enriched.json"
    )

    profiles_by_id = {
        profile["profile_id"]: profile
        for profile in profiles
    }

    jobs_by_id = {
        job["id"]: job
        for job in jobs
    }

    split_files = {
        "train":
            SPLIT_DIR / "train_labels.json",
        "validation":
            SPLIT_DIR / "validation_labels.json",
        "test":
            SPLIT_DIR / "test_labels.json",
        "gold":
            SPLIT_DIR / "gold_evaluation.json",
    }

    split_reports = {}

    for split_name, split_path in (
        split_files.items()
    ):
        labels = load_json(split_path)

        split_reports[split_name] = (
            evaluate_split(
                split_name,
                labels,
                profiles_by_id,
                jobs_by_id,
            )
        )

    final_report = {
        "model": {
            "name":
                "Keyword and skill coverage baseline",
            "type":
                "deterministic non-trained baseline",
        },
        "weights": {
            "candidate_primary":
                CANDIDATE_PRIMARY_WEIGHT,
            "candidate_secondary":
                CANDIDATE_SECONDARY_WEIGHT,
            "job_required":
                JOB_REQUIRED_WEIGHT,
            "job_preferred":
                JOB_PREFERRED_WEIGHT,
            "job_mentioned":
                JOB_MENTIONED_WEIGHT,
        },
        "splits": split_reports,
        "interpretation": {
            "primary_target":
                "skills_score",
            "secondary_target":
                "composite_score",
            "score_calibration":
                "No artificial score offset or cap",
            "gold_limit":
                "Benchmark and excluded low-quality jobs "
                "without enriched features are reported "
                "as unavailable.",
        },
        "safety_rules": {
            "generated_matches_used":
                False,
            "generated_gaps_used":
                False,
            "generated_scores_used_as_inputs":
                False,
            "gold_used_for_weight_tuning":
                False,
            "artificial_score_offset":
                False,
            "artificial_score_cap":
                False,
        },
    }

    save_json(
        OUTPUT_DIR
        / "keyword_baseline_report.json",
        final_report,
    )

    print(
        "Stage 6 keyword baseline evaluation completed."
    )
    print("")

    for split_name, report in (
        split_reports.items()
    ):
        print(
            f"{split_name.title()}: "
            f"{report['available_scores']:,}/"
            f"{report['input_labels']:,} "
            f"pairs evaluated"
        )

        print(
            "  Skills MAE: "
            f"{report['skills_target_metrics']['mae']}"
        )

        print(
            "  Skills Spearman: "
            f"{report['skills_target_metrics']['spearman']}"
        )

        print(
            "  Skills NDCG@10: "
            f"{report['skills_target_ranking']['ndcg_at_10']['mean']}"
        )

    print("")
    print(
        "Report created at: "
        f"{OUTPUT_DIR / 'keyword_baseline_report.json'}"
    )


if __name__ == "__main__":
    main()