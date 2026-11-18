import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MIN_DESCRIPTION_LENGTH = 100


def load_json(filename):
    with (RAW_DIR / filename).open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json(filename, data):
    output_path = PROCESSED_DIR / filename

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_pair_id(pair_id):
    profile_part, separator, job_part = pair_id.partition("_")

    if not separator:
        return None, None

    try:
        profile_id = int(profile_part)
    except ValueError:
        return None, None

    return profile_id, job_part


def deduplicate_jobs(jobs):
    unique_jobs = {}
    identical_duplicates_removed = 0

    for job in jobs:
        job_id = job["id"]

        if job_id not in unique_jobs:
            unique_jobs[job_id] = job
            continue

        if unique_jobs[job_id] != job:
            raise ValueError(
                f"Conflicting duplicate job detected: {job_id}"
            )

        identical_duplicates_removed += 1

    return list(unique_jobs.values()), identical_duplicates_removed


def prepare_profiles(profiles):
    prepared_profiles = []

    for profile in profiles:
        cleaned_profile = dict(profile)

        # According to the dataset documentation, the 438 profiles
        # without a data_origin value are synthetic profiles.
        if not cleaned_profile.get("data_origin"):
            cleaned_profile["data_origin"] = "synthetic"

        prepared_profiles.append(cleaned_profile)

    return prepared_profiles


def prepare_training_labels(
    labels,
    valid_profile_ids,
    valid_job_ids,
):
    cleaned_labels = []

    excluded_invalid_format = 0
    excluded_unknown_profile = 0
    excluded_unknown_job = 0

    for label in labels:
        profile_id, job_id = parse_pair_id(label["pair_id"])

        if profile_id is None or job_id is None:
            excluded_invalid_format += 1
            continue

        if profile_id not in valid_profile_ids:
            excluded_unknown_profile += 1
            continue

        if job_id not in valid_job_ids:
            excluded_unknown_job += 1
            continue

        cleaned_labels.append(
            {
                "pair_id": label["pair_id"],
                "profile_id": profile_id,
                "job_id": job_id,
                "skills_score": label["skills"],
                "seniority_score": label["seniority"],
                "domain_score": label["domain"],
                "location_score": label["location"],
                "composite_score": label["composite"],
                "overqualified": label["overqualified"],
                "label_source": label["label_source"],
                "confidence": label["confidence"],
            }
        )

    exclusions = {
        "invalid_pair_id_format": excluded_invalid_format,
        "unknown_profile": excluded_unknown_profile,
        "unknown_or_excluded_job": excluded_unknown_job,
    }

    return cleaned_labels, exclusions


def prepare_baseline_pairs(
    pairs,
    cleaned_labels,
):
    labels_by_pair_id = {
        label["pair_id"]: label
        for label in cleaned_labels
    }

    baseline_pairs = []
    missing_label_count = 0

    for pair in pairs:
        pair_id = pair["pair_id"]
        label = labels_by_pair_id.get(pair_id)

        if label is None:
            missing_label_count += 1
            continue

        baseline_pairs.append(
            {
                "pair_id": pair_id,
                "profile_id": pair["profile_id"],
                "job_id": pair["job_id"],
                "pair_type": pair["pair_type"],
                "profile_role_family":
                    pair["profile_role_family"],
                "job_role_family":
                    pair["job_role_family"],
                "profile_seniority":
                    pair["profile_seniority"],

                # This field is preserved only for comparison with
                # the dataset's original baseline. It must not be
                # supplied to the GNN as an input feature.
                "baseline_only_skill_overlap":
                    pair["skill_overlap"],

                "skills_score": label["skills_score"],
                "seniority_score": label["seniority_score"],
                "domain_score": label["domain_score"],
                "location_score": label["location_score"],
                "composite_score": label["composite_score"],
                "overqualified": label["overqualified"],
            }
        )

    return baseline_pairs, missing_label_count


def prepare_gold_evaluation(
    gold_labels,
    profile_ids,
    main_jobs_by_id,
    all_main_jobs_by_id,
    benchmark_jobs_by_id,
):
    gold_evaluation = []

    excluded_invalid_format = 0
    excluded_unknown_profile = 0
    excluded_unknown_job = 0

    for gold in gold_labels:
        profile_id, job_id = parse_pair_id(gold["pair_id"])

        if profile_id is None or job_id is None:
            excluded_invalid_format += 1
            continue

        if profile_id not in profile_ids:
            excluded_unknown_profile += 1
            continue

        if job_id in main_jobs_by_id:
            job_source = "main_dataset"
            low_quality_job = False

        elif job_id in all_main_jobs_by_id:
            job_source = "excluded_low_quality_main_job"
            low_quality_job = True

        elif job_id in benchmark_jobs_by_id:
            job_source = "benchmark_dataset"
            low_quality_job = False

        else:
            excluded_unknown_job += 1
            continue

        cleaned_gold = {
            "pair_id": gold["pair_id"],
            "profile_id": profile_id,
            "job_id": job_id,
            "job_source": job_source,
            "low_quality_job": low_quality_job,
            "skills_score": gold["skills"],
            "seniority_score": gold["seniority"],
            "domain_score": gold["domain"],
            "location_score": gold["location"],
            "composite_score": gold["composite"],
            "overqualified": gold["overqualified"],
        }

        optional_fields = [
            "reasoning",
            "matches",
            "gaps",
            "job_family",
            "strategy",
            "review_status",
        ]

        for field in optional_fields:
            if field in gold:
                cleaned_gold[field] = gold[field]

        gold_evaluation.append(cleaned_gold)

    exclusions = {
        "invalid_pair_id_format": excluded_invalid_format,
        "unknown_profile": excluded_unknown_profile,
        "unknown_job": excluded_unknown_job,
    }

    return gold_evaluation, exclusions


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_jobs = load_json("scraped_jobs.json")
    raw_profiles = load_json("synthetic_profiles.json")
    raw_pairs = load_json("phase3_pairs.json")
    raw_labels = load_json("phase3_labels.json")
    raw_gold_labels = load_json("gold_labels.json")
    benchmarks = load_json("benchmarks.json")

    unique_jobs, duplicate_jobs_removed = deduplicate_jobs(
        raw_jobs
    )

    excluded_short_jobs = [
        {
            "id": job["id"],
            "title": job["title"],
            "description_length":
                len(job["description"].strip()),
        }
        for job in unique_jobs
        if len(job["description"].strip())
        < MIN_DESCRIPTION_LENGTH
    ]

    excluded_short_job_ids = {
        job["id"]
        for job in excluded_short_jobs
    }

    cleaned_jobs = [
        job
        for job in unique_jobs
        if job["id"] not in excluded_short_job_ids
    ]

    cleaned_profiles = prepare_profiles(raw_profiles)

    profile_ids = {
        profile["profile_id"]
        for profile in cleaned_profiles
    }

    valid_job_ids = {
        job["id"]
        for job in cleaned_jobs
    }

    main_jobs_by_id = {
        job["id"]: job
        for job in cleaned_jobs
    }
    all_main_jobs_by_id = {
        job["id"]: job
        for job in unique_jobs
    }

    benchmark_jobs = benchmarks.get("jobs", [])

    benchmark_jobs_by_id = {
        job["id"]: job
        for job in benchmark_jobs
    }

    cleaned_labels, label_exclusions = (
        prepare_training_labels(
            raw_labels,
            profile_ids,
            valid_job_ids,
        )
    )

    baseline_pairs, missing_baseline_labels = (
        prepare_baseline_pairs(
            raw_pairs,
            cleaned_labels,
        )
    )

    gold_evaluation, gold_exclusions = (
        prepare_gold_evaluation(
            raw_gold_labels,
            profile_ids,
            main_jobs_by_id,
            all_main_jobs_by_id,
            benchmark_jobs_by_id,
        )
    )

    save_json("jobs_clean.json", cleaned_jobs)
    save_json("profiles_clean.json", cleaned_profiles)
    save_json("training_labels_clean.json", cleaned_labels)
    save_json("baseline_pairs_clean.json", baseline_pairs)
    save_json("gold_evaluation.json", gold_evaluation)
    save_json("benchmark_jobs.json", benchmark_jobs)

    manifest = {
        "raw_counts": {
            "jobs": len(raw_jobs),
            "profiles": len(raw_profiles),
            "phase3_pairs": len(raw_pairs),
            "generated_labels": len(raw_labels),
            "gold_labels": len(raw_gold_labels),
        },
        "processed_counts": {
            "jobs": len(cleaned_jobs),
            "profiles": len(cleaned_profiles),
            "training_labels": len(cleaned_labels),
            "baseline_pairs": len(baseline_pairs),
            "gold_evaluation": len(gold_evaluation),
            "benchmark_jobs": len(benchmark_jobs),
        },
        "cleaning_actions": {
            "identical_duplicate_jobs_removed":
                duplicate_jobs_removed,
            "short_description_jobs_removed":
                len(excluded_short_jobs),
            "short_description_jobs":
                excluded_short_jobs,
            "generated_label_exclusions":
                label_exclusions,
            "baseline_pairs_without_usable_labels":
                missing_baseline_labels,
            "gold_label_exclusions":
                gold_exclusions,
        },
        "model_safety_rules": {
            "matches_and_gaps_used_as_inputs": False,
            "label_component_scores_used_as_inputs": False,
            "baseline_skill_overlap_used_as_gnn_input": False,
            "gold_labels_reserved_for_final_evaluation": True,
        },
    }

    save_json("preprocessing_manifest.json", manifest)

    print("Role Radar preprocessing completed successfully.")
    print("")
    print(f"Clean jobs: {len(cleaned_jobs):,}")
    print(f"Clean profiles: {len(cleaned_profiles):,}")
    print(f"Clean training labels: {len(cleaned_labels):,}")
    print(f"Clean baseline pairs: {len(baseline_pairs):,}")
    print(f"Gold evaluation pairs: {len(gold_evaluation):,}")
    print("")
    print(
        "Manifest created at: "
        f"{PROCESSED_DIR / 'preprocessing_manifest.json'}"
    )


if __name__ == "__main__":
    main()