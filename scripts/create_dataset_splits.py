import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def load_json(filename):
    with (PROCESSED_DIR / filename).open(
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


def primary_role(profile):
    roles = profile.get(
        "derived_role_families",
        [],
    )

    if roles:
        return roles[0]

    return "unknown"


def profile_origin(profile):
    return profile.get(
        "data_origin",
        "unknown",
    )


def allocate_role_group(
    profile_ids,
    profiles_by_id,
    random_generator,
):
    """
    Splits one role family while interleaving profile origins.

    This helps avoid putting all public or synthetic profiles
    into only one split.
    """

    origin_groups = defaultdict(list)

    for profile_id in profile_ids:
        origin = profile_origin(
            profiles_by_id[profile_id]
        )

        origin_groups[origin].append(
            profile_id
        )

    for ids in origin_groups.values():
        random_generator.shuffle(ids)

    interleaved_ids = []
    ordered_origins = sorted(
        origin_groups
    )

    while any(
        origin_groups[origin]
        for origin in ordered_origins
    ):
        for origin in ordered_origins:
            if origin_groups[origin]:
                interleaved_ids.append(
                    origin_groups[origin].pop()
                )

    count = len(interleaved_ids)

    if count == 1:
        return interleaved_ids, [], []

    if count == 2:
        return (
            [interleaved_ids[0]],
            [interleaved_ids[1]],
            [],
        )

    validation_count = max(
        1,
        round(count * VALIDATION_RATIO),
    )

    test_count = max(
        1,
        round(count * TEST_RATIO),
    )

    while (
        validation_count + test_count
        >= count
    ):
        if test_count >= validation_count:
            test_count -= 1
        else:
            validation_count -= 1

    training_count = (
        count
        - validation_count
        - test_count
    )

    training_ids = interleaved_ids[
        :training_count
    ]

    validation_ids = interleaved_ids[
        training_count:
        training_count + validation_count
    ]

    test_ids = interleaved_ids[
        training_count + validation_count:
    ]

    return (
        training_ids,
        validation_ids,
        test_ids,
    )


def score_statistics(labels):
    scores = [
        label["composite_score"]
        for label in labels
    ]

    if not scores:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "mean": round(
            statistics.mean(scores),
            3,
        ),
        "median": round(
            statistics.median(scores),
            3,
        ),
    }


def candidate_role_distribution(
    candidate_ids,
    profiles_by_id,
):
    counter = Counter(
        primary_role(
            profiles_by_id[candidate_id]
        )
        for candidate_id in candidate_ids
    )

    return dict(counter.most_common())


def candidate_origin_distribution(
    candidate_ids,
    profiles_by_id,
):
    counter = Counter(
        profile_origin(
            profiles_by_id[candidate_id]
        )
        for candidate_id in candidate_ids
    )

    return dict(counter.most_common())


def split_summary(
    candidate_ids,
    labels,
    profiles_by_id,
):
    return {
        "candidate_count":
            len(candidate_ids),
        "label_count":
            len(labels),
        "unique_job_count":
            len(
                {
                    label["job_id"]
                    for label in labels
                }
            ),
        "role_distribution":
            candidate_role_distribution(
                candidate_ids,
                profiles_by_id,
            ),
        "origin_distribution":
            candidate_origin_distribution(
                candidate_ids,
                profiles_by_id,
            ),
        "composite_score_statistics":
            score_statistics(labels),
    }


def main():
    profiles = load_json(
        "profiles_features_enriched.json"
    )

    jobs = load_json(
        "jobs_features_enriched.json"
    )

    training_labels = load_json(
        "training_labels_clean.json"
    )

    gold_evaluation = load_json(
        "gold_evaluation.json"
    )

    profiles_by_id = {
        profile["profile_id"]: profile
        for profile in profiles
    }

    all_profile_ids = set(
        profiles_by_id
    )

    generated_label_candidate_ids = {
        label["profile_id"]
        for label in training_labels
    }

    gold_candidate_ids = {
        label["profile_id"]
        for label in gold_evaluation
    }

    gold_pair_ids = {
        label["pair_id"]
        for label in gold_evaluation
    }

    # Every candidate used in gold evaluation is held out.
    eligible_candidate_ids = (
        generated_label_candidate_ids
        - gold_candidate_ids
    )

    profiles_without_generated_labels = (
        all_profile_ids
        - generated_label_candidate_ids
    )

    reserved_gold_generated_labels = [
        label
        for label in training_labels
        if label["profile_id"]
        in gold_candidate_ids
    ]

    eligible_labels = [
        label
        for label in training_labels
        if label["profile_id"]
        in eligible_candidate_ids
        and label["pair_id"]
        not in gold_pair_ids
    ]

    candidates_by_role = defaultdict(list)

    for candidate_id in eligible_candidate_ids:
        role = primary_role(
            profiles_by_id[candidate_id]
        )

        candidates_by_role[role].append(
            candidate_id
        )

    random_generator = random.Random(
        RANDOM_SEED
    )

    train_candidate_ids = []
    validation_candidate_ids = []
    test_candidate_ids = []

    for role in sorted(candidates_by_role):
        (
            role_train_ids,
            role_validation_ids,
            role_test_ids,
        ) = allocate_role_group(
            candidates_by_role[role],
            profiles_by_id,
            random_generator,
        )

        train_candidate_ids.extend(
            role_train_ids
        )

        validation_candidate_ids.extend(
            role_validation_ids
        )

        test_candidate_ids.extend(
            role_test_ids
        )

    train_candidate_set = set(
        train_candidate_ids
    )

    validation_candidate_set = set(
        validation_candidate_ids
    )

    test_candidate_set = set(
        test_candidate_ids
    )

    # Leakage-safety assertions.
    assert train_candidate_set.isdisjoint(
        validation_candidate_set
    )

    assert train_candidate_set.isdisjoint(
        test_candidate_set
    )

    assert validation_candidate_set.isdisjoint(
        test_candidate_set
    )

    assert train_candidate_set.isdisjoint(
        gold_candidate_ids
    )

    assert validation_candidate_set.isdisjoint(
        gold_candidate_ids
    )

    assert test_candidate_set.isdisjoint(
        gold_candidate_ids
    )

    assert (
        train_candidate_set
        | validation_candidate_set
        | test_candidate_set
    ) == eligible_candidate_ids

    train_labels = [
        label
        for label in eligible_labels
        if label["profile_id"]
        in train_candidate_set
    ]

    validation_labels = [
        label
        for label in eligible_labels
        if label["profile_id"]
        in validation_candidate_set
    ]

    test_labels = [
        label
        for label in eligible_labels
        if label["profile_id"]
        in test_candidate_set
    ]

    train_pair_ids = {
        label["pair_id"]
        for label in train_labels
    }

    validation_pair_ids = {
        label["pair_id"]
        for label in validation_labels
    }

    test_pair_ids = {
        label["pair_id"]
        for label in test_labels
    }

    assert train_pair_ids.isdisjoint(
        validation_pair_ids
    )

    assert train_pair_ids.isdisjoint(
        test_pair_ids
    )

    assert validation_pair_ids.isdisjoint(
        test_pair_ids
    )

    assert train_pair_ids.isdisjoint(
        gold_pair_ids
    )

    assert validation_pair_ids.isdisjoint(
        gold_pair_ids
    )

    assert test_pair_ids.isdisjoint(
        gold_pair_ids
    )

    # Sort outputs to make them reproducible.
    train_candidate_ids = sorted(
        train_candidate_ids
    )

    validation_candidate_ids = sorted(
        validation_candidate_ids
    )

    test_candidate_ids = sorted(
        test_candidate_ids
    )

    train_labels = sorted(
        train_labels,
        key=lambda item: item["pair_id"],
    )

    validation_labels = sorted(
        validation_labels,
        key=lambda item: item["pair_id"],
    )

    test_labels = sorted(
        test_labels,
        key=lambda item: item["pair_id"],
    )

    save_json(
        SPLIT_DIR / "train_candidate_ids.json",
        train_candidate_ids,
    )

    save_json(
        SPLIT_DIR / "validation_candidate_ids.json",
        validation_candidate_ids,
    )

    save_json(
        SPLIT_DIR / "test_candidate_ids.json",
        test_candidate_ids,
    )

    save_json(
        SPLIT_DIR / "train_labels.json",
        train_labels,
    )

    save_json(
        SPLIT_DIR / "validation_labels.json",
        validation_labels,
    )

    save_json(
        SPLIT_DIR / "test_labels.json",
        test_labels,
    )

    save_json(
        SPLIT_DIR / "gold_evaluation.json",
        gold_evaluation,
    )

    save_json(
        SPLIT_DIR
        / "reserved_gold_candidate_generated_labels.json",
        reserved_gold_generated_labels,
    )

    save_json(
        SPLIT_DIR
        / "profiles_without_generated_labels.json",
        sorted(
            profiles_without_generated_labels
        ),
    )

    manifest = {
        "configuration": {
            "random_seed": RANDOM_SEED,
            "split_strategy":
                "candidate-grouped cold-start split",
            "training_target_ratio": 0.70,
            "validation_target_ratio": 0.15,
            "test_target_ratio": 0.15,
            "gold_candidates_reserved": True,
            "stratification":
                "primary role with origin interleaving",
        },
        "source_counts": {
            "jobs": len(jobs),
            "profiles": len(profiles),
            "generated_labels":
                len(training_labels),
            "gold_labels":
                len(gold_evaluation),
            "generated_label_candidates":
                len(
                    generated_label_candidate_ids
                ),
            "gold_candidates":
                len(gold_candidate_ids),
            "profiles_without_generated_labels":
                len(
                    profiles_without_generated_labels
                ),
            "generated_labels_reserved_due_to_gold_candidate":
                len(
                    reserved_gold_generated_labels
                ),
        },
        "splits": {
            "train": split_summary(
                train_candidate_ids,
                train_labels,
                profiles_by_id,
            ),
            "validation": split_summary(
                validation_candidate_ids,
                validation_labels,
                profiles_by_id,
            ),
            "test": split_summary(
                test_candidate_ids,
                test_labels,
                profiles_by_id,
            ),
            "gold": {
                "candidate_count":
                    len(gold_candidate_ids),
                "label_count":
                    len(gold_evaluation),
                "main_dataset_jobs":
                    sum(
                        label.get("job_source")
                        == "main_dataset"
                        for label in gold_evaluation
                    ),
                "benchmark_dataset_jobs":
                    sum(
                        label.get("job_source")
                        == "benchmark_dataset"
                        for label in gold_evaluation
                    ),
                "low_quality_jobs":
                    sum(
                        label.get(
                            "low_quality_job",
                            False,
                        )
                        for label in gold_evaluation
                    ),
            },
        },
        "leakage_checks": {
            "train_validation_candidate_overlap":
                len(
                    train_candidate_set
                    & validation_candidate_set
                ),
            "train_test_candidate_overlap":
                len(
                    train_candidate_set
                    & test_candidate_set
                ),
            "validation_test_candidate_overlap":
                len(
                    validation_candidate_set
                    & test_candidate_set
                ),
            "train_gold_candidate_overlap":
                len(
                    train_candidate_set
                    & gold_candidate_ids
                ),
            "validation_gold_candidate_overlap":
                len(
                    validation_candidate_set
                    & gold_candidate_ids
                ),
            "test_gold_candidate_overlap":
                len(
                    test_candidate_set
                    & gold_candidate_ids
                ),
            "train_gold_pair_overlap":
                len(
                    train_pair_ids
                    & gold_pair_ids
                ),
            "validation_gold_pair_overlap":
                len(
                    validation_pair_ids
                    & gold_pair_ids
                ),
            "test_gold_pair_overlap":
                len(
                    test_pair_ids
                    & gold_pair_ids
                ),
        },
        "graph_training_rules": {
            "train_candidate_nodes_only_during_training":
                True,
            "validation_candidates_added_only_during_validation":
                True,
            "test_candidates_added_only_during_testing":
                True,
            "gold_candidates_never_used_for_training":
                True,
            "all_job_features_available":
                True,
            "gold_labels_used_for_model_selection":
                False,
        },
    }

    save_json(
        SPLIT_DIR / "split_manifest.json",
        manifest,
    )

    print("Stage 5 dataset splitting completed.")
    print("")
    print(
        f"Training candidates: "
        f"{len(train_candidate_ids):,}"
    )
    print(
        f"Validation candidates: "
        f"{len(validation_candidate_ids):,}"
    )
    print(
        f"Test candidates: "
        f"{len(test_candidate_ids):,}"
    )
    print(
        f"Reserved gold candidates: "
        f"{len(gold_candidate_ids):,}"
    )
    print("")
    print(
        f"Training labels: "
        f"{len(train_labels):,}"
    )
    print(
        f"Validation labels: "
        f"{len(validation_labels):,}"
    )
    print(
        f"Test labels: "
        f"{len(test_labels):,}"
    )
    print("")
    print(
        "Manifest created at: "
        f"{SPLIT_DIR / 'split_manifest.json'}"
    )


if __name__ == "__main__":
    main()