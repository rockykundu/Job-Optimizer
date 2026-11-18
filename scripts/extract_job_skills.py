import html
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


PREFERRED_HEADINGS = (
    "preferred qualifications",
    "preferred skills",
    "nice to have",
    "nice-to-have",
    "good to have",
    "desirable",
    "bonus skills",
    "bonus points",
    "additional qualifications",
)

REQUIRED_HEADINGS = (
    "requirements",
    "required qualifications",
    "required skills",
    "minimum qualifications",
    "must have",
    "must-have",
    "qualifications",
    "what you need",
    "what we are looking for",
    "what we're looking for",
    "skills required",
    "skills",
    "technical skills",
    "key skills",
    "core skills",
    "essential skills",
    "skills and experience",
    "skills & experience",
)

GENERAL_HEADINGS = (
    "responsibilities",
    "key responsibilities",
    "job responsibilities",
    "roles and responsibilities",
    "duties",
    "essential functions",
    "job description",
    "job summary",
    "role overview",
    "about the role",
    "about us",
    "company overview",
    "what you will do",
    "what you'll do",
    "what we offer",
    "benefits",
    "perks",
    "compensation",
    "compensation and benefits",
    "why join us",
    "why work with us",
    "equal opportunity",
    "equal employment opportunity",
    "working at",
)

def load_json(filename):
    with (PROCESSED_DIR / filename).open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json(filename, data):
    with (PROCESSED_DIR / filename).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def clean_text(value):
    if not isinstance(value, str):
        return ""

    text = html.unescape(value)
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text


def create_lookup_key(value):
    text = clean_text(value).casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip(" \t\r\n,;:|•·")


def clean_heading(line):
    line = create_lookup_key(line)
    line = re.sub(r"^[\-*•\d.)\s]+", "", line)

    return line.strip(" :.-")


def heading_matches(line, headings):
    cleaned = clean_heading(line)

    return any(
        cleaned == heading
        or cleaned.startswith(heading + ":")
        for heading in headings
    )


def content_after_heading(line):
    if ":" not in line:
        return ""

    return line.split(":", 1)[1].strip()


def split_description_sections(description):
    """
    Divides a description into general, required and preferred
    sections without using any generated labels.
    """

    sections = {
        "general": [],
        "required": [],
        "preferred": [],
    }

    current_section = "general"

    for original_line in clean_text(description).split("\n"):
        line = original_line.strip()

        if not line:
            continue

        if heading_matches(line, PREFERRED_HEADINGS):
            current_section = "preferred"

            remainder = content_after_heading(line)

            if remainder:
                sections[current_section].append(remainder)

            continue

        if heading_matches(line, REQUIRED_HEADINGS):
            current_section = "required"

            remainder = content_after_heading(line)

            if remainder:
                sections[current_section].append(remainder)

            continue

        if heading_matches(line, GENERAL_HEADINGS):
            current_section = "general"

            remainder = content_after_heading(line)

            if remainder:
                sections[current_section].append(remainder)

            continue

        sections[current_section].append(line)

    return {
        section: "\n".join(lines)
        for section, lines in sections.items()
    }


def build_skill_dictionary(skill_catalog, alias_data):
    phrase_to_canonical = {}
    conflicts = defaultdict(set)

    # These require special context handling because ordinary
    # case-insensitive matching would produce false positives.
    ambiguous_short_skills = {
        "r",
        "c",
        "go",

        # These are valid in specific technical contexts but create
        # false positives when matched as ordinary English words.
        "room",
        "provider",
        "rest",
        "restful",
        "express",
    }

    def register_phrase(phrase, canonical):
        phrase_key = create_lookup_key(phrase)

        if not phrase_key:
            return

        if phrase_key in ambiguous_short_skills:
            return

        existing = phrase_to_canonical.get(phrase_key)

        if existing and existing != canonical:
            conflicts[phrase_key].update(
                [existing, canonical]
            )
            return

        phrase_to_canonical[phrase_key] = canonical

    canonical_names = set()

    for catalog_entry in skill_catalog:
        canonical = catalog_entry["canonical_name"]
        canonical_names.add(canonical)

        register_phrase(canonical, canonical)

        for variant in catalog_entry.get("raw_variants", []):
            register_phrase(
                variant["value"],
                canonical,
            )

    for alias, canonical in alias_data["aliases"].items():
        register_phrase(alias, canonical)
        register_phrase(canonical, canonical)
        canonical_names.add(canonical)

    phrases = sorted(
        phrase_to_canonical,
        key=len,
        reverse=True,
    )

    escaped_phrases = [
        re.escape(phrase)
        for phrase in phrases
    ]

    if escaped_phrases:
        combined_pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:"
            + "|".join(escaped_phrases)
            + r")(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
    else:
        combined_pattern = None

    canonical_lookup = {
        canonical.casefold(): canonical
        for canonical in canonical_names
    }

    return (
        phrase_to_canonical,
        combined_pattern,
        canonical_lookup,
        conflicts,
    )


def extract_context_sensitive_skills(
    text,
    canonical_lookup,
):
    detected = set()

    # R programming language
    if "r" in canonical_lookup:
        r_patterns = (
            r"\bR programming\b",
            r"\bR language\b",
            r"\bPython\s*[/,;&]\s*R\b",
            r"\bR\s*[/,;&]\s*Python\b",
            r"\bexperience (?:with|in) R\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in r_patterns
        ):
            detected.add(canonical_lookup["r"])

    # C programming language
    if "c" in canonical_lookup:
        c_patterns = (
            r"\bC programming\b",
            r"\bC language\b",
            r"\bembedded C\b",
            r"\bC\s*[/,;&]\s*C\+\+\b",
            r"\bC\+\+\s*[/,;&]\s*C\b",
            r"\bexperience (?:with|in) C\b",
        )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in c_patterns
        ):
            detected.add(canonical_lookup["c"])

    # Go programming language
    if "go" in canonical_lookup:
        go_patterns = (
            r"\bGolang\b",
            r"\bGo programming\b",
            r"\bGo language\b",
            r"\bGo developer\b",
            r"\bGo development\b",
            r"\bGo engineer\b",
            r"\bexperience (?:with|in) Go\b",
        )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in go_patterns
        ):
            detected.add(canonical_lookup["go"])
            
    # REST API: require uppercase REST or explicit RESTful API wording.
    rest_canonical = canonical_lookup.get("rest apis")

    if rest_canonical:
        rest_patterns = (
            r"\bREST APIs?\b",
            r"\bRESTful APIs?\b",
            r"\bREST architecture\b",
            r"\bREST services?\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in rest_patterns
        ):
            detected.add(rest_canonical)

    # Express.js: require framework-specific context.
    express_canonical = canonical_lookup.get("express.js")

    if express_canonical:
        express_patterns = (
            r"\bExpress\.js\b",
            r"\bExpressJS\b",
            r"\bExpress framework\b",
            r"\bNode\.js\s+(?:with|and)\s+Express\b",
            r"\bNodeJS\s+(?:with|and)\s+Express\b",
        )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in express_patterns
        ):
            detected.add(express_canonical)

    return detected


def extract_skills(
    text,
    phrase_to_canonical,
    combined_pattern,
    canonical_lookup,
):
    cleaned_text = clean_text(text)
    detected = set()

    if combined_pattern is not None:
        for match in combined_pattern.finditer(
            cleaned_text.casefold()
        ):
            matched_phrase = create_lookup_key(
                match.group(0)
            )

            canonical = phrase_to_canonical.get(
                matched_phrase
            )

            if canonical:
                detected.add(canonical)

    detected.update(
        extract_context_sensitive_skills(
            cleaned_text,
            canonical_lookup,
        )
    )

    return detected


def sort_skills(skills):
    return sorted(
        skills,
        key=lambda value: value.casefold(),
    )


def main():
    jobs = load_json("jobs_clean.json")
    skill_catalog = load_json("skill_catalog.json")
    alias_data = load_json("skill_aliases.json")

    (
        phrase_to_canonical,
        combined_pattern,
        canonical_lookup,
        conflicts,
    ) = build_skill_dictionary(
        skill_catalog,
        alias_data,
    )

    enriched_jobs = []
    jobs_without_skills = []
    all_skill_counter = Counter()
    required_counter = Counter()
    preferred_counter = Counter()
    general_counter = Counter()
    role_statistics = defaultdict(
        lambda: {
            "jobs": 0,
            "jobs_with_skills": 0,
            "total_skills": 0,
        }
    )

    jobs_with_required_sections = 0
    jobs_with_preferred_sections = 0

    for job in jobs:
        description = job.get("description", "")
        title = job.get("title", "")

        sections = split_description_sections(
            description
        )

        general_text = (
            title
            + "\n"
            + sections["general"]
        )

        general_skills = extract_skills(
            general_text,
            phrase_to_canonical,
            combined_pattern,
            canonical_lookup,
        )

        required_skills = extract_skills(
            sections["required"],
            phrase_to_canonical,
            combined_pattern,
            canonical_lookup,
        )

        preferred_skills = extract_skills(
            sections["preferred"],
            phrase_to_canonical,
            combined_pattern,
            canonical_lookup,
        )

        if sections["required"].strip():
            jobs_with_required_sections += 1

        if sections["preferred"].strip():
            jobs_with_preferred_sections += 1

        # If a skill appears in a required section, required wins.
        preferred_skills -= required_skills
        general_skills -= required_skills
        general_skills -= preferred_skills

        all_skills = (
            required_skills
            | preferred_skills
            | general_skills
        )

        for skill in all_skills:
            all_skill_counter[skill] += 1

        for skill in required_skills:
            required_counter[skill] += 1

        for skill in preferred_skills:
            preferred_counter[skill] += 1

        for skill in general_skills:
            general_counter[skill] += 1

        role_family = job.get(
            "role_family_hint",
            "unknown",
        )

        role_statistics[role_family]["jobs"] += 1
        role_statistics[role_family][
            "total_skills"
        ] += len(all_skills)

        if all_skills:
            role_statistics[role_family][
                "jobs_with_skills"
            ] += 1
        else:
            jobs_without_skills.append(
                {
                    "id": job["id"],
                    "title": title,
                    "role_family": role_family,
                    "description_length":
                        len(description),
                }
            )

        enriched_job = dict(job)

        enriched_job["skills_required"] = sort_skills(
            required_skills
        )

        enriched_job["skills_preferred"] = sort_skills(
            preferred_skills
        )

        enriched_job["skills_mentioned"] = sort_skills(
            general_skills
        )

        enriched_job["skills_all"] = sort_skills(
            all_skills
        )

        enriched_jobs.append(enriched_job)

    skill_counts_per_job = [
        len(job["skills_all"])
        for job in enriched_jobs
    ]

    role_coverage = []

    for role_family, values in sorted(
        role_statistics.items()
    ):
        job_count = values["jobs"]
        jobs_with_skills = values["jobs_with_skills"]

        role_coverage.append(
            {
                "role_family": role_family,
                "job_count": job_count,
                "jobs_with_skills":
                    jobs_with_skills,
                "jobs_without_skills":
                    job_count - jobs_with_skills,
                "coverage_percentage": round(
                    (
                        jobs_with_skills
                        / job_count
                        * 100
                    )
                    if job_count
                    else 0,
                    2,
                ),
                "average_skills_per_job": round(
                    (
                        values["total_skills"]
                        / job_count
                    )
                    if job_count
                    else 0,
                    2,
                ),
            }
        )

    report = {
        "input": {
            "job_count": len(jobs),
            "candidate_skill_catalog_size":
                len(skill_catalog),
            "registered_search_phrases":
                len(phrase_to_canonical),
        },
        "output": {
            "enriched_job_count":
                len(enriched_jobs),
            "jobs_with_at_least_one_skill":
                len(enriched_jobs)
                - len(jobs_without_skills),
            "jobs_without_detected_skills":
                len(jobs_without_skills),
            "unique_detected_skills":
                len(all_skill_counter),
            "required_sections_detected":
                jobs_with_required_sections,
            "preferred_sections_detected":
                jobs_with_preferred_sections,
            "average_skills_per_job": round(
                statistics.mean(
                    skill_counts_per_job
                ),
                2,
            ),
            "median_skills_per_job": round(
                statistics.median(
                    skill_counts_per_job
                ),
                2,
            ),
            "maximum_skills_in_one_job":
                max(skill_counts_per_job),
        },
        "top_detected_skills": [
            {
                "skill": skill,
                "job_count": count,
                "required_count":
                    required_counter[skill],
                "preferred_count":
                    preferred_counter[skill],
                "general_mention_count":
                    general_counter[skill],
            }
            for skill, count in
            all_skill_counter.most_common(50)
        ],
        "role_family_coverage":
            role_coverage,
        "quality_review": {
            "dictionary_conflict_count":
                len(conflicts),
            "dictionary_conflicts": {
                phrase: sorted(values)
                for phrase, values in conflicts.items()
            },
            "jobs_without_skills_sample":
                jobs_without_skills[:50],
        },
        "safety_rules": {
            "generated_matches_used": False,
            "generated_gaps_used": False,
            "generated_scores_used": False,
            "job_descriptions_used": True,
            "candidate_skill_catalog_used": True,
            "ambiguous_short_skills_context_checked":
                True,
            "original_job_records_preserved": True,
        },
    }

    # Deterministic review sample: first three jobs per role.
    review_sample = []
    role_review_counts = Counter()

    for job in sorted(
        enriched_jobs,
        key=lambda item: (
            item.get("role_family_hint", ""),
            item["id"],
        ),
    ):
        role = job.get(
            "role_family_hint",
            "unknown",
        )

        if role_review_counts[role] >= 3:
            continue

        review_sample.append(
            {
                "id": job["id"],
                "title": job["title"],
                "role_family": role,
                "description_excerpt":
                    job["description"][:600],
                "skills_required":
                    job["skills_required"],
                "skills_preferred":
                    job["skills_preferred"],
                "skills_mentioned":
                    job["skills_mentioned"],
            }
        )

        role_review_counts[role] += 1

    save_json(
        "jobs_skills_extracted.json",
        enriched_jobs,
    )

    save_json(
        "job_skill_extraction_report.json",
        report,
    )

    save_json(
        "job_skill_extraction_review.json",
        review_sample,
    )

    print("Stage 4A job-skill extraction completed.")
    print("")
    print(f"Jobs processed: {len(enriched_jobs):,}")
    print(
        "Jobs with detected skills: "
        f"{report['output']['jobs_with_at_least_one_skill']:,}"
    )
    print(
        "Jobs without detected skills: "
        f"{len(jobs_without_skills):,}"
    )
    print(
        "Unique detected skills: "
        f"{len(all_skill_counter):,}"
    )
    print(
        "Average skills per job: "
        f"{report['output']['average_skills_per_job']}"
    )
    print("")
    print(
        "Report created at: "
        f"{PROCESSED_DIR / 'job_skill_extraction_report.json'}"
    )


if __name__ == "__main__":
    main()