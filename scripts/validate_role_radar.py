import json
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
REPORT_PATH = AUDIT_DIR / "role_radar_integrity.txt"


def load_json(filename):
    path = RAW_DIR / filename

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def duplicate_count(values):
    values = list(values)
    return len(values) - len(set(values))


def parse_pair_id(pair_id):
    """
    Expected format:
    profileID_linkedin_jobID
    """

    first_part, separator, remaining = pair_id.partition("_")

    if not separator:
        return None, None

    try:
        profile_id = int(first_part)
    except ValueError:
        return None, None

    return profile_id, remaining


def describe_numbers(values):
    values = [value for value in values if isinstance(value, (int, float))]

    if not values:
        return "No numeric values"

    return (
        f"count={len(values):,}, "
        f"minimum={min(values):.2f}, "
        f"maximum={max(values):.2f}, "
        f"mean={statistics.mean(values):.2f}, "
        f"median={statistics.median(values):.2f}, "
        f"unique_values={len(set(values)):,}"
    )


def add_counter(report, title, values):
    report.append(title)

    counter = Counter(values)

    for value, count in counter.most_common():
        report.append(f"  - {value}: {count:,}")


def main():
    jobs = load_json("scraped_jobs.json")
    profiles = load_json("synthetic_profiles.json")
    pairs = load_json("phase3_pairs.json")
    labels = load_json("phase3_labels.json")
    gold_labels = load_json("gold_labels.json")
    benchmarks = load_json("benchmarks.json")

    report = [
        "ROLE RADAR DATASET — INTEGRITY AND JOIN AUDIT",
        "=" * 80,
        "",
    ]

    job_ids = [job["id"] for job in jobs]
    profile_ids = [profile["profile_id"] for profile in profiles]
    pair_ids = [pair["pair_id"] for pair in pairs]
    label_pair_ids = [label["pair_id"] for label in labels]
    gold_pair_ids = [label["pair_id"] for label in gold_labels]

    job_id_set = set(job_ids)
    profile_id_set = set(profile_ids)
    pair_id_set = set(pair_ids)
    label_pair_id_set = set(label_pair_ids)
    gold_pair_id_set = set(gold_pair_ids)

    report.extend(
        [
            "1. PRIMARY COUNTS",
            f"Jobs: {len(jobs):,}",
            f"Profiles: {len(profiles):,}",
            f"Phase 3 pairs: {len(pairs):,}",
            f"Phase 3 labels: {len(labels):,}",
            f"Gold labels: {len(gold_labels):,}",
            "",
            "2. DUPLICATE IDENTIFIERS",
            f"Duplicate job IDs: {duplicate_count(job_ids):,}",
            f"Duplicate profile IDs: {duplicate_count(profile_ids):,}",
            f"Duplicate pair IDs: {duplicate_count(pair_ids):,}",
            f"Duplicate label pair IDs: "
            f"{duplicate_count(label_pair_ids):,}",
            f"Duplicate gold pair IDs: "
            f"{duplicate_count(gold_pair_ids):,}",
            "",
        ]
    )

    # Validate phase3_pairs.json foreign keys.
    invalid_pair_profiles = [
        pair
        for pair in pairs
        if pair["profile_id"] not in profile_id_set
    ]

    invalid_pair_jobs = [
        pair
        for pair in pairs
        if pair["job_id"] not in job_id_set
    ]

    report.extend(
        [
            "3. PHASE 3 PAIR FOREIGN KEYS",
            f"Pairs with unknown profile IDs: "
            f"{len(invalid_pair_profiles):,}",
            f"Pairs with unknown job IDs: "
            f"{len(invalid_pair_jobs):,}",
            "",
        ]
    )

    # Reconstruct profile and job IDs from every label pair_id.
    unresolved_label_format = 0
    unknown_label_profiles = 0
    unknown_label_jobs = 0

    for label in labels:
        profile_id, job_id = parse_pair_id(label["pair_id"])

        if profile_id is None or job_id is None:
            unresolved_label_format += 1
            continue

        if profile_id not in profile_id_set:
            unknown_label_profiles += 1

        if job_id not in job_id_set:
            unknown_label_jobs += 1

    report.extend(
        [
            "4. LABEL IDENTIFIER RECONSTRUCTION",
            f"Labels with invalid pair_id format: "
            f"{unresolved_label_format:,}",
            f"Labels referencing unknown profiles: "
            f"{unknown_label_profiles:,}",
            f"Labels referencing unknown jobs: "
            f"{unknown_label_jobs:,}",
            "",
            "5. PAIR/LABEL COVERAGE",
            f"Phase 3 pairs also present in labels: "
            f"{len(pair_id_set & label_pair_id_set):,}",
            f"Phase 3 pairs missing labels: "
            f"{len(pair_id_set - label_pair_id_set):,}",
            f"Labels not present in phase3_pairs.json: "
            f"{len(label_pair_id_set - pair_id_set):,}",
            f"Gold labels also present in generated labels: "
            f"{len(gold_pair_id_set & label_pair_id_set):,}",
            "",
        ]
    )

    # Validate gold foreign keys.
    invalid_gold_format = 0
    unknown_gold_profiles = 0
    unknown_gold_jobs = 0

    for label in gold_labels:
        profile_id, job_id = parse_pair_id(label["pair_id"])

        if profile_id is None or job_id is None:
            invalid_gold_format += 1
            continue

        if profile_id not in profile_id_set:
            unknown_gold_profiles += 1

        if job_id not in job_id_set:
            unknown_gold_jobs += 1

    report.extend(
        [
            "6. GOLD-LABEL IDENTIFIERS",
            f"Gold labels with invalid pair_id format: "
            f"{invalid_gold_format:,}",
            f"Gold labels referencing unknown profiles: "
            f"{unknown_gold_profiles:,}",
            f"Gold labels referencing unknown jobs: "
            f"{unknown_gold_jobs:,}",
            "",
        ]
    )

    # Numerical label distributions.
    report.append("7. GENERATED LABEL DISTRIBUTIONS")

    for field in [
        "skills",
        "seniority",
        "domain",
        "location",
        "composite",
    ]:
        values = [label[field] for label in labels]
        report.append(f"{field}: {describe_numbers(values)}")

    report.append("")
    report.append("8. GOLD LABEL DISTRIBUTIONS")

    for field in [
        "skills",
        "seniority",
        "domain",
        "location",
        "composite",
    ]:
        values = [label[field] for label in gold_labels]
        report.append(f"{field}: {describe_numbers(values)}")

    report.append("")

    add_counter(
        report,
        "9. GENERATED LABEL SOURCES",
        [label["label_source"] for label in labels],
    )

    report.append("")

    add_counter(
        report,
        "10. GENERATED LABEL CONFIDENCE",
        [label["confidence"] for label in labels],
    )

    report.append("")

    add_counter(
        report,
        "11. PAIR TYPES",
        [pair["pair_type"] for pair in pairs],
    )

    report.append("")

    add_counter(
        report,
        "12. JOB ROLE FAMILIES",
        [job["role_family_hint"] for job in jobs],
    )

    report.append("")

    add_counter(
        report,
        "13. JOB SENIORITY LEVELS",
        [job["seniority_level"] for job in jobs],
    )

    report.append("")

    add_counter(
        report,
        "14. PROFILE SENIORITY LEVELS",
        [profile["seniority"] for profile in profiles],
    )

    report.append("")

    # Skill statistics.
    primary_skills = [
        skill.strip()
        for profile in profiles
        for skill in profile["skills_primary"]
        if isinstance(skill, str) and skill.strip()
    ]

    secondary_skills = [
        skill.strip()
        for profile in profiles
        for skill in profile["skills_secondary"]
        if isinstance(skill, str) and skill.strip()
    ]

    normalised_skills = {
        skill.casefold()
        for skill in primary_skills + secondary_skills
    }

    report.extend(
        [
            "15. PROFILE SKILLS",
            f"Primary skill occurrences: {len(primary_skills):,}",
            f"Secondary skill occurrences: {len(secondary_skills):,}",
            f"Unique case-insensitive skills: "
            f"{len(normalised_skills):,}",
            "",
        ]
    )

    # Job-description checks.
    description_lengths = [
        len(job["description"].strip())
        for job in jobs
        if isinstance(job.get("description"), str)
    ]

    empty_descriptions = sum(
        not isinstance(job.get("description"), str)
        or not job["description"].strip()
        for job in jobs
    )

    report.extend(
        [
            "16. JOB DESCRIPTIONS",
            f"Empty job descriptions: {empty_descriptions:,}",
            f"Description character lengths: "
            f"{describe_numbers(description_lengths)}",
            "",
        ]
    )

    # Determine variations in gold-label schemas.
    gold_schema_patterns = Counter(
        tuple(sorted(label.keys()))
        for label in gold_labels
    )

    report.append("17. GOLD-LABEL SCHEMA PATTERNS")
    report.append(
        f"Distinct gold-label structures: "
        f"{len(gold_schema_patterns):,}"
    )

    for schema, count in gold_schema_patterns.most_common():
        report.append(f"  - Records: {count:,}")
        report.append(f"    Fields: {list(schema)}")

    report.append("")

    # Compare generated and gold scores where pair IDs overlap.
    generated_by_pair = {
        label["pair_id"]: label
        for label in labels
    }

    common_gold_labels = [
        label
        for label in gold_labels
        if label["pair_id"] in generated_by_pair
    ]

    report.append("18. GENERATED VERSUS GOLD SCORE COMPARISON")
    report.append(
        f"Comparable overlapping pairs: "
        f"{len(common_gold_labels):,}"
    )

    for field in [
        "skills",
        "seniority",
        "domain",
        "location",
        "composite",
    ]:
        differences = [
            abs(
                gold[field]
                - generated_by_pair[gold["pair_id"]][field]
            )
            for gold in common_gold_labels
        ]

        exact_matches = sum(difference == 0 for difference in differences)

        if differences:
            report.append(
                f"{field}: exact={exact_matches:,}/"
                f"{len(differences):,}, "
                f"mean_absolute_difference="
                f"{statistics.mean(differences):.2f}, "
                f"maximum_difference={max(differences):.2f}"
            )

    report.append("")
    report.append("19. BENCHMARK DATASET")
    report.append(
        f"Benchmark profile available: "
        f"{'benchmark_profile' in benchmarks}"
    )
    report.append(
        f"Benchmark jobs: "
        f"{len(benchmarks.get('jobs', [])):,}"
    )
    report.append(
        f"Benchmark baseline records: "
        f"{len(benchmarks.get('baselines', [])):,}"
    )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("Role Radar integrity audit completed.")
    print(f"Report created at: {REPORT_PATH}")


if __name__ == "__main__":
    main()