import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
REPORT_PATH = AUDIT_DIR / "role_radar_anomalies.txt"


def load_json(filename):
    with (RAW_DIR / filename).open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def parse_pair_id(pair_id):
    profile_part, separator, job_part = pair_id.partition("_")

    if not separator:
        return None, None

    try:
        profile_id = int(profile_part)
    except ValueError:
        return None, None

    return profile_id, job_part


def differing_fields(records):
    if len(records) < 2:
        return []

    fields = set()

    for record in records:
        fields.update(record.keys())

    first_record = records[0]

    return sorted(
        field
        for field in fields
        if any(
            record.get(field) != first_record.get(field)
            for record in records[1:]
        )
    )


def main():
    jobs = load_json("scraped_jobs.json")
    profiles = load_json("synthetic_profiles.json")
    pairs = load_json("phase3_pairs.json")
    labels = load_json("phase3_labels.json")
    gold_labels = load_json("gold_labels.json")

    report = [
        "ROLE RADAR — TARGETED ANOMALY DIAGNOSTIC",
        "=" * 80,
        "",
    ]

    jobs_by_id = defaultdict(list)

    for job in jobs:
        jobs_by_id[job["id"]].append(job)

    duplicate_groups = {
        job_id: records
        for job_id, records in jobs_by_id.items()
        if len(records) > 1
    }

    report.append("1. DUPLICATE JOB ANALYSIS")
    report.append(
        f"Unique job IDs: {len(jobs_by_id):,}"
    )
    report.append(
        f"Duplicate job-ID groups: {len(duplicate_groups):,}"
    )
    report.append(
        f"Extra duplicated rows: "
        f"{sum(len(records) - 1 for records in duplicate_groups.values()):,}"
    )

    identical_groups = 0
    conflicting_groups = 0

    for job_id, records in sorted(duplicate_groups.items()):
        differences = differing_fields(records)

        if differences:
            conflicting_groups += 1
            status = "CONFLICTING"
        else:
            identical_groups += 1
            status = "IDENTICAL"

        report.append(
            f"  - {job_id}: rows={len(records)}, "
            f"status={status}, "
            f"differing_fields={differences}"
        )

    report.append(
        f"Identical duplicate groups: {identical_groups:,}"
    )
    report.append(
        f"Conflicting duplicate groups: {conflicting_groups:,}"
    )
    report.append("")

    profile_ids = {
        profile["profile_id"]
        for profile in profiles
    }

    job_ids = set(jobs_by_id)

    unknown_profile_counter = Counter()
    unknown_job_counter = Counter()
    valid_label_pairs = []

    for label in labels:
        profile_id, job_id = parse_pair_id(label["pair_id"])

        if profile_id not in profile_ids:
            unknown_profile_counter[profile_id] += 1

        if job_id not in job_ids:
            unknown_job_counter[job_id] += 1

        if profile_id in profile_ids and job_id in job_ids:
            valid_label_pairs.append(label)

    report.append("2. GENERATED-LABEL FOREIGN KEYS")
    report.append(
        f"Usable labels: {len(valid_label_pairs):,}"
    )
    report.append(
        f"Excluded labels: {len(labels) - len(valid_label_pairs):,}"
    )
    report.append(
        f"Distinct unknown profile IDs: "
        f"{len(unknown_profile_counter):,}"
    )
    report.append(
        f"Distinct unknown job IDs: "
        f"{len(unknown_job_counter):,}"
    )

    report.append("Unknown profile IDs and label counts:")

    for profile_id, count in unknown_profile_counter.most_common():
        report.append(
            f"  - profile_id={profile_id}: labels={count:,}"
        )

    if unknown_job_counter:
        report.append("Unknown job IDs and label counts:")

        for job_id, count in unknown_job_counter.most_common():
            report.append(
                f"  - job_id={job_id}: labels={count:,}"
            )

    report.append("")

    valid_candidate_ids = set()
    valid_job_ids = set()

    for label in valid_label_pairs:
        profile_id, job_id = parse_pair_id(label["pair_id"])
        valid_candidate_ids.add(profile_id)
        valid_job_ids.add(job_id)

    report.append("3. USABLE LABEL COVERAGE")
    report.append(
        f"Profiles represented: "
        f"{len(valid_candidate_ids):,}/{len(profile_ids):,}"
    )
    report.append(
        f"Jobs represented: "
        f"{len(valid_job_ids):,}/{len(job_ids):,}"
    )
    report.append("")

    pair_ids = {
        pair["pair_id"]
        for pair in pairs
    }

    label_ids = {
        label["pair_id"]
        for label in labels
    }

    missing_pair_labels = sorted(pair_ids - label_ids)

    report.append("4. PHASE 3 PAIRS WITHOUT LABELS")
    report.append(
        f"Missing-label pairs: {len(missing_pair_labels):,}"
    )

    for pair_id in missing_pair_labels:
        report.append(f"  - {pair_id}")

    report.append("")

    pair_type_by_id = {
        pair["pair_id"]: pair["pair_type"]
        for pair in pairs
    }

    labelled_pair_types = Counter(
        pair_type_by_id[pair_id]
        for pair_id in pair_ids & label_ids
    )

    report.append("5. LABELLED PHASE 3 PAIR TYPES")

    for pair_type, count in labelled_pair_types.most_common():
        report.append(f"  - {pair_type}: {count:,}")

    report.append("")

    unknown_gold_jobs = []
    unknown_gold_profiles = []

    for gold in gold_labels:
        profile_id, job_id = parse_pair_id(gold["pair_id"])

        if profile_id not in profile_ids:
            unknown_gold_profiles.append(gold["pair_id"])

        if job_id not in job_ids:
            unknown_gold_jobs.append(gold["pair_id"])

    report.append("6. GOLD-LABEL ANOMALIES")
    report.append(
        f"Unknown-profile gold labels: "
        f"{len(unknown_gold_profiles):,}"
    )

    for pair_id in unknown_gold_profiles:
        report.append(f"  - {pair_id}")

    report.append(
        f"Unknown-job gold labels: "
        f"{len(unknown_gold_jobs):,}"
    )

    for pair_id in unknown_gold_jobs:
        report.append(f"  - {pair_id}")

    report.append("")

    short_descriptions = [
        (
            job["id"],
            len(job["description"].strip()),
            job["title"],
        )
        for job in jobs
        if len(job["description"].strip()) < 100
    ]

    report.append("7. VERY SHORT JOB DESCRIPTIONS")
    report.append(
        f"Descriptions under 100 characters: "
        f"{len(short_descriptions):,}"
    )

    for job_id, length, title in short_descriptions:
        report.append(
            f"  - {job_id}: characters={length}, "
            f"title={title}"
        )

    report.append("")

    origin_counter = Counter(
        profile.get("data_origin", "not_provided")
        for profile in profiles
    )

    report.append("8. PROFILE DATA ORIGINS")

    for origin, count in origin_counter.most_common():
        report.append(f"  - {origin}: {count:,}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("Targeted anomaly diagnostic completed.")
    print(f"Report created at: {REPORT_PATH}")


if __name__ == "__main__":
    main()