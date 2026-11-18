import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Only high-confidence aliases are included.
# Uncertain skills are kept separate instead of being incorrectly merged.
HIGH_CONFIDENCE_ALIASES = {
    # JavaScript ecosystem
    "node": "Node.js",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",

    "react": "React",
    "react js": "React",
    "reactjs": "React",
    "react.js": "React",

    "vue": "Vue.js",
    "vue js": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",

    "next js": "Next.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",

    "javascript": "JavaScript",
    "java script": "JavaScript",
    "js": "JavaScript",

    "typescript": "TypeScript",
    "type script": "TypeScript",
    "ts": "TypeScript",

    # Cloud and infrastructure
    "amazon web services": "AWS",
    "aws": "AWS",

    "google cloud platform": "GCP",
    "google cloud": "GCP",
    "gcp": "GCP",

    "microsoft azure": "Azure",
    "azure": "Azure",

    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",

    "ci cd": "CI/CD",
    "ci/cd": "CI/CD",
    "continuous integration continuous deployment": "CI/CD",
    "continuous integration/continuous deployment": "CI/CD",

    # Databases
    "postgres": "PostgreSQL",
    "postgre sql": "PostgreSQL",
    "postgresql": "PostgreSQL",

    "mongo db": "MongoDB",
    "mongodb": "MongoDB",

    "my sql": "MySQL",
    "mysql": "MySQL",

    "ms sql": "Microsoft SQL Server",
    "mssql": "Microsoft SQL Server",
    "microsoft sql server": "Microsoft SQL Server",

    # Programming languages and frameworks
    "c sharp": "C#",
    "c-sharp": "C#",
    "c#": "C#",

    "c plus plus": "C++",
    "c-plus-plus": "C++",
    "c++": "C++",

    "dot net": ".NET",
    "dotnet": ".NET",
    ".net": ".NET",

    "objective c": "Objective-C",
    "objective-c": "Objective-C",

    "springboot": "Spring Boot",
    "spring boot": "Spring Boot",

    # Machine learning and data
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",

    "artificial intelligence": "Artificial Intelligence",
    "ai": "Artificial Intelligence",

    "natural language processing": "Natural Language Processing",
    "nlp": "Natural Language Processing",

    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",

    "tensor flow": "TensorFlow",
    "tensorflow": "TensorFlow",

    "py torch": "PyTorch",
    "pytorch": "PyTorch",

    "powerbi": "Power BI",
    "power bi": "Power BI",

    # APIs and version control
    "rest": "REST APIs",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "restful api": "REST APIs",
    "restful apis": "REST APIs",

    "graphql": "GraphQL",

    "git hub": "GitHub",
    "github": "GitHub",

    # Business tools
    "communication": "Communication",
    "communication skill": "Communication",
    "communication skills": "Communication",

    "excel": "Microsoft Excel",
    "ms excel": "Microsoft Excel",
    "microsoft excel": "Microsoft Excel",

    "search engine optimization": "SEO",
    "search engine optimisation": "SEO",
    "seo": "SEO",

    "customer relationship management": "CRM",
    "crm": "CRM",

    # Additional confirmed aliases
    "golang": "Go",
    "go programming": "Go",
    "go language": "Go",

    "ms office": "Microsoft Office",
    "ms office suite": "Microsoft Office",
    "microsoft office": "Microsoft Office",
    "microsoft office suite": "Microsoft Office",

    "onboarding process": "Onboarding",

    "cloud infrastructure": "Cloud Infrastructure",

    "distributed system": "Distributed Systems",
    "distributed systems": "Distributed Systems",

    "express": "Express.js",
    "express js": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
}


def load_json(filename):
    file_path = PROCESSED_DIR / filename

    with file_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json(filename, data):
    file_path = PROCESSED_DIR / filename

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def clean_surface_text(value):
    if not isinstance(value, str):
        return ""

    text = unicodedata.normalize("NFKC", value)

    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip(" \t\r\n,;:|•·")


def create_lookup_key(value):
    text = clean_surface_text(value).casefold()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


ALIAS_LOOKUP = {
    create_lookup_key(alias): canonical
    for alias, canonical in HIGH_CONFIDENCE_ALIASES.items()
}


def get_group_key(raw_skill):
    cleaned = clean_surface_text(raw_skill)
    lookup_key = create_lookup_key(cleaned)

    if lookup_key in ALIAS_LOOKUP:
        canonical = ALIAS_LOOKUP[lookup_key]
        return f"alias::{canonical.casefold()}"

    return f"raw::{lookup_key}"


def get_alias_canonical(raw_skill):
    lookup_key = create_lookup_key(raw_skill)
    return ALIAS_LOOKUP.get(lookup_key)


def unique_preserving_order(values):
    seen = set()
    output = []

    for value in values:
        comparison_key = value.casefold()

        if comparison_key in seen:
            continue

        seen.add(comparison_key)
        output.append(value)

    return output


def main():
    profiles = load_json("profiles_clean.json")
    manifest = load_json("preprocessing_manifest.json")

    gold_count = (
        manifest
        .get("processed_counts", {})
        .get("gold_evaluation")
    )

    warnings = []

    if gold_count != 108:
        warnings.append(
            "The preprocessing manifest does not contain "
            "108 gold evaluation records."
        )

    group_variants = defaultdict(Counter)
    group_primary_count = Counter()
    group_secondary_count = Counter()
    group_profile_ids = defaultdict(set)
    alias_occurrence_count = 0
    empty_skills_removed = 0

    # First pass: collect every skill variation.
    for profile in profiles:
        profile_id = profile["profile_id"]

        for skill in profile.get("skills_primary", []):
            cleaned = clean_surface_text(skill)

            if not cleaned:
                empty_skills_removed += 1
                continue

            group_key = get_group_key(cleaned)
            group_variants[group_key][cleaned] += 1
            group_primary_count[group_key] += 1
            group_profile_ids[group_key].add(profile_id)

            if get_alias_canonical(cleaned):
                alias_occurrence_count += 1

        for skill in profile.get("skills_secondary", []):
            cleaned = clean_surface_text(skill)

            if not cleaned:
                empty_skills_removed += 1
                continue

            group_key = get_group_key(cleaned)
            group_variants[group_key][cleaned] += 1
            group_secondary_count[group_key] += 1
            group_profile_ids[group_key].add(profile_id)

            if get_alias_canonical(cleaned):
                alias_occurrence_count += 1

    # Determine one canonical display name for each group.
    canonical_by_group = {}

    for group_key, variants in group_variants.items():
        if group_key.startswith("alias::"):
            first_variant = variants.most_common(1)[0][0]
            canonical = get_alias_canonical(first_variant)

            if canonical is None:
                canonical = first_variant
        else:
            # Use the most common original spelling when no explicit
            # alias exists.
            canonical = variants.most_common(1)[0][0]

        canonical_by_group[group_key] = canonical

    normalized_profiles = []
    primary_duplicates_removed = 0
    secondary_duplicates_removed = 0
    primary_secondary_overlaps_removed = 0

    # Second pass: apply canonical names to every profile.
    for profile in profiles:
        original_primary = profile.get("skills_primary", [])
        original_secondary = profile.get("skills_secondary", [])

        normalized_primary_before_deduplication = []

        for skill in original_primary:
            cleaned = clean_surface_text(skill)

            if not cleaned:
                continue

            group_key = get_group_key(cleaned)
            normalized_primary_before_deduplication.append(
                canonical_by_group[group_key]
            )

        normalized_primary = unique_preserving_order(
            normalized_primary_before_deduplication
        )

        primary_duplicates_removed += (
            len(normalized_primary_before_deduplication)
            - len(normalized_primary)
        )

        normalized_secondary_before_deduplication = []

        for skill in original_secondary:
            cleaned = clean_surface_text(skill)

            if not cleaned:
                continue

            group_key = get_group_key(cleaned)
            normalized_secondary_before_deduplication.append(
                canonical_by_group[group_key]
            )

        normalized_secondary = unique_preserving_order(
            normalized_secondary_before_deduplication
        )

        secondary_duplicates_removed += (
            len(normalized_secondary_before_deduplication)
            - len(normalized_secondary)
        )

        primary_keys = {
            skill.casefold()
            for skill in normalized_primary
        }

        filtered_secondary = [
            skill
            for skill in normalized_secondary
            if skill.casefold() not in primary_keys
        ]

        primary_secondary_overlaps_removed += (
            len(normalized_secondary)
            - len(filtered_secondary)
        )

        normalized_profile = dict(profile)

        # Preserve original values for traceability.
        normalized_profile["skills_primary_raw"] = original_primary
        normalized_profile["skills_secondary_raw"] = (
            original_secondary
        )

        normalized_profile["skills_primary"] = normalized_primary
        normalized_profile["skills_secondary"] = filtered_secondary

        normalized_profiles.append(normalized_profile)

    skill_catalog = []

    for group_key, canonical in sorted(
        canonical_by_group.items(),
        key=lambda item: item[1].casefold(),
    ):
        variants = group_variants[group_key]

        skill_catalog.append(
            {
                "canonical_name": canonical,
                "group_key": group_key,
                "total_occurrences": sum(variants.values()),
                "primary_occurrences":
                    group_primary_count[group_key],
                "secondary_occurrences":
                    group_secondary_count[group_key],
                "profile_count":
                    len(group_profile_ids[group_key]),
                "raw_variants": [
                    {
                        "value": value,
                        "count": count,
                    }
                    for value, count in variants.most_common()
                ],
                "explicit_alias_group":
                    group_key.startswith("alias::"),
            }
        )

    suspicious_skills = []

    for item in skill_catalog:
        skill = item["canonical_name"]

        reasons = []

        if len(skill) == 1:
            reasons.append("single_character")

        if len(skill) > 80:
            reasons.append("unusually_long")

        if skill.isdigit():
            reasons.append("numeric_only")

        if "\n" in skill or "\r" in skill:
            reasons.append("contains_line_break")

        if reasons:
            suspicious_skills.append(
                {
                    "skill": skill,
                    "reasons": reasons,
                    "occurrences": item["total_occurrences"],
                }
            )

    alias_file = {
        "description": (
            "Conservative, high-confidence skill aliases used "
            "for candidate-profile normalisation."
        ),
        "aliases": dict(
            sorted(
                HIGH_CONFIDENCE_ALIASES.items(),
                key=lambda item: item[0].casefold(),
            )
        ),
    }

    report = {
        "input": {
            "profile_count": len(profiles),
            "raw_primary_occurrences": sum(
                len(profile.get("skills_primary", []))
                for profile in profiles
            ),
            "raw_secondary_occurrences": sum(
                len(profile.get("skills_secondary", []))
                for profile in profiles
            ),
            "raw_unique_case_insensitive_skills": len(
                {
                    clean_surface_text(skill).casefold()
                    for profile in profiles
                    for skill in (
                        profile.get("skills_primary", [])
                        + profile.get("skills_secondary", [])
                    )
                    if clean_surface_text(skill)
                }
            ),
        },
        "output": {
            "normalized_profile_count":
                len(normalized_profiles),
            "canonical_skill_count":
                len(skill_catalog),
            "normalized_primary_occurrences": sum(
                len(profile["skills_primary"])
                for profile in normalized_profiles
            ),
            "normalized_secondary_occurrences": sum(
                len(profile["skills_secondary"])
                for profile in normalized_profiles
            ),
        },
        "normalization_actions": {
            "alias_occurrences_processed":
                alias_occurrence_count,
            "empty_skills_removed":
                empty_skills_removed,
            "primary_duplicates_removed":
                primary_duplicates_removed,
            "secondary_duplicates_removed":
                secondary_duplicates_removed,
            "primary_secondary_overlaps_removed":
                primary_secondary_overlaps_removed,
        },
        "quality_review": {
            "suspicious_skill_count":
                len(suspicious_skills),
            "suspicious_skills":
                suspicious_skills,
            "warnings":
                warnings,
        },
        "safety_rules": {
            "original_skills_preserved": True,
            "uncertain_aliases_automatically_merged": False,
            "generated_label_matches_used": False,
            "generated_label_gaps_used": False,
            "job_skills_extracted_during_this_stage": False,
        },
    }

    save_json(
        "profiles_skills_normalized.json",
        normalized_profiles,
    )

    save_json(
        "skill_catalog.json",
        skill_catalog,
    )

    save_json(
        "skill_aliases.json",
        alias_file,
    )

    save_json(
        "skill_normalization_report.json",
        report,
    )

    print("Stage 3 skill normalisation completed.")
    print("")
    print(
        f"Profiles processed: "
        f"{len(normalized_profiles):,}"
    )
    print(
        f"Canonical skills: "
        f"{len(skill_catalog):,}"
    )
    print(
        f"Alias occurrences processed: "
        f"{alias_occurrence_count:,}"
    )
    print(
        f"Primary/secondary overlaps removed: "
        f"{primary_secondary_overlaps_removed:,}"
    )
    print("")
    print(
        "Report created at: "
        f"{PROCESSED_DIR / 'skill_normalization_report.json'}"
    )

    if warnings:
        print("")
        print("Warnings:")

        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()