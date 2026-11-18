import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


ROLE_RULES = [
    (
        "hr",
        [
            r"\bhuman resources?\b",
            r"\bhr\b",
            r"\brecruit(?:er|ing|ment)\b",
            r"\btalent acquisition\b",
            r"\btalent development\b",
            r"\bpeople operations\b",
            r"\bemployee relations\b",
            r"\bworkforce\b",
            r"\blearning\s*(?:&|and)\s*development\b",
            r"\btraining manager\b",
            r"\borganizational development\b",
            r"\bpayroll\b",
            r"\btalent partner\b",
            r"\btalent vp\b",
            r"\bcompensation manager\b",
        ],
    ),
    (
        "legal",
        [
            r"\blegal\b",
            r"\blawyer\b",
            r"\battorney\b",
            r"\bcounsel\b",
            r"\bparalegal\b",
            r"\bfinancial crime compliance\b",
            r"\baml\b",
            r"\bkyc\b",
            r"\bregulatory affairs\b",
            r"\bcompliance (?:officer|analyst|manager)\b",
            r"\bcontract manager\b",
            r"\brisk manager\b",
        ],
    ),
    (
        "finance",
        [
            r"\bfinance\b",
            r"\bfinancial analyst\b",
            r"\baccountant\b",
            r"\baccounting\b",
            r"\bchartered accountant\b",
            r"\bauditor\b",
            r"\baudit manager\b",
            r"\btax\b",
            r"\btreasury\b",
            r"\bcredit analyst\b",
            r"\binvestment\b",
            r"\breconciliation\b",
            r"\bfp&a\b",
            r"\bfinancial planning\s*(?:&|and)\s*analysis\b",
            r"\bcontroller\b",
            r"\baccounts payable\b",
            r"\baccounts receivable\b",
        ],
    ),
    (
        "marketing",
        [
            r"\bmarketing\b",
            r"\bdigital marketing\b",
            r"\bcontent marketing\b",
            r"\bseo\b",
            r"\bsem\b",
            r"\bbrand manager\b",
            r"\bsocial media\b",
            r"\bgraphic design\b",
            r"\bcampaign manager\b",
            r"\bmarket research\b",
            r"\bcontent writer\b",
            r"\be-commerce specialist\b",
            r"\becommerce specialist\b",
            r"\bgrowth manager\b",
            r"\bgraphic designer\b",
        ],
    ),
    (
        "sales",
        [
            r"\bsales\b",
            r"\bbusiness development\b",
            r"\baccount executive\b",
            r"\bsales development\b",
            r"\bmedical representative\b",
            r"\bterritory manager\b",
            r"\bkey account manager\b",
            r"\baccount manager\b",
            r"\bclient partner\b",
        ],
    ),
    (
        "customer_success",
        [
            r"\bcustomer success\b",
            r"\bcustomer service\b",
            r"\bcustomer support\b",
            r"\bclient success\b",
            r"\brelationship manager\b",
            r"\bservice delivery\b",
            r"\bclient relationship\b",
            r"\bimplementation manager\b",
            r"\bcustomer experience\b",
            r"\bcustomer onboarding\b",
            r"\bsupport manager\b",
            r"\bcs director\b",
        ],
    ),
    (
        "operations",
        [
            r"\boperations?\b",
            r"\bsupply chain\b",
            r"\blogistics\b",
            r"\bprocurement\b",
            r"\bwarehouse\b",
            r"\binventory\b",
            r"\bfacilities\b",
            r"\bpractice manager\b",
            r"\bnurse manager\b",
            r"\bmaintenance manager\b",
            r"\bproduction supervisor\b",
            r"\bprocess improvement\b",
        ],
    ),
    (
        "product",
        [
            r"\bproduct manager\b",
            r"\bproduct owner\b",
            r"\bproduct lead\b",
            r"\bproduct analyst\b",
            r"\bprogram manager\b",
            r"\bproject manager\b",
            r"\bdelivery lead\b",
            r"\bbusiness analyst\b",
            r"\bproduct engineer\b",
            r"\btechnical pm\b",
            r"\bprogram director\b",
            r"\bstaff pm\b",
            r"\bdesign engineer\b",
        ],
    ),
    (
        "mobile",
        [
            r"\bandroid\b",
            r"\bios developer\b",
            r"\bmobile developer\b",
            r"\bmobile engineer\b",
            r"\bflutter developer\b",
            r"\breact native developer\b",
            r"\bmobile app developer\b",
            r"\bios engineer\b",
            r"\bmobile architect\b",
            r"\bflutter architect\b",
            r"\bar/vr developer\b",
            r"\bgame developer\b",
        ],
    ),
    (
        "devops",
        [
            r"\bdevops\b",
            r"\bsite reliability\b",
            r"\bsre\b",
            r"\bcloud engineer\b",
            r"\bcloud architect\b",
            r"\binfrastructure engineer\b",
            r"\bplatform engineer\b",
            r"\bsystems administrator\b",
            r"\bsystem administrator\b",
            r"\bdatabase administrator\b",
            r"\bit support\b",
            r"\bsupport engineer\b",
            r"\bnetwork engineer\b",
            r"\bcybersecurity\b",
            r"\bplatform architect\b",
        ],
    ),
    (
        "data",
        [
            r"\bdata scientist\b",
            r"\bdata analyst\b",
            r"\bdata engineer\b",
            r"\bmachine learning\b",
            r"\bml engineer\b",
            r"\bai engineer\b",
            r"\bbusiness intelligence\b",
            r"\bbi developer\b",
            r"\banalytics engineer\b",
            r"\bai researcher\b",
            r"\bdata architect\b",
        ],
    ),
    (
        "frontend",
        [
            r"\bfront[\s-]?end\b",
            r"\bui developer\b",
            r"\bweb designer\b",
            r"\breact developer\b",
            r"\bangular developer\b",
            r"\bvue developer\b",
            r"\bui engineer\b",
            r"\bui designer\b",
            r"\bux designer\b",
            r"\bui/ux designer\b",
            r"\bui\s*/\s*ux\b",
            r"\bui architect\b",
        ],
    ),
    (
        "backend",
        [
            r"\bback[\s-]?end\b",
            r"\bapi developer\b",
            r"\bjava developer\b",
            r"\bpython developer\b",
            r"\bphp developer\b",
            r"\bnode(?:\.js)? developer\b",
            r"\bserver[\s-]?side\b",
            r"\b\.net developer\b",
            r"\bruby developer\b",
            r"\bjava software engineer\b",
            r"\bpython software engineer\b",
            r"\bnode(?:\.js)? software engineer\b",
            r"\bbackend engineer\b",
            r"\bbackend developer\b",
            r"\bapi engineer\b",
            r"\bapi architect\b",
            r"\bblockchain developer\b",
            r"\bgo developer\b",
            r"\bmicroservices engineer\b",

            # These patterns must beat the generic Software Engineer rule.
            r"\bsoftware engineer.*back[\s-]?end\b",
            r"\bsoftware engineer.*backend\b",
        ],
    ),
    (
        "fullstack",
        [
            r"\bfull[\s-]?stack\b",
            r"\bsoftware engineer\b",
            r"\bsoftware developer\b",
            r"\bapplication developer\b",
            r"\bweb developer\b",
            r"\btest engineer\b",
            r"\bqa engineer\b",
            r"\brobotics engineer\b",
        ],
    ),
]


INDIAN_CITY_NAMES = {
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "noida": "Noida",
    "gurugram": "Gurugram",
    "gurgaon": "Gurugram",
    "hyderabad": "Hyderabad",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "lucknow": "Lucknow",
    "chandigarh": "Chandigarh",
    "indore": "Indore",
    "surat": "Surat",
    "vadodara": "Vadodara",
    "baroda": "Vadodara",
    "goa": "Goa",
}
US_STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de",
    "fl", "ga", "hi", "id", "il", "in", "ia", "ks",
    "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny",
    "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc",
}

SOURCE_SENIORITY_MAPPING = {
    "internship": "Intern",
    "entry level": "Junior",
    "associate": "Mid",
    "mid-senior level": "Senior",
    "director": "Staff",
    "executive": "Staff",
}


SENIORITY_ORDINAL = {
    "Unknown": -1,
    "Intern": 0,
    "Junior": 1,
    "Mid": 2,
    "Senior": 3,
    "Staff": 4,
}


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


def normalise_text(value):
    if not isinstance(value, str):
        return ""

    return re.sub(r"\s+", " ", value).strip()


def derive_role_family(title, fallback=None):
    """
    Selects the earliest and most specific role phrase in the title.

    This prevents a later domain phrase such as 'Marketing Team' or
    'Supply Chain' from overriding an earlier occupational title such
    as 'Data Engineer' or 'Backend Engineer'.
    """

    title_text = normalise_text(title).casefold()
    matches = []

    for family_index, (role_family, patterns) in enumerate(
        ROLE_RULES
    ):
        for pattern_index, pattern in enumerate(patterns):
            match = re.search(
                pattern,
                title_text,
                flags=re.IGNORECASE,
            )

            if match:
                matched_length = match.end() - match.start()

                matches.append(
                    {
                        "role_family": role_family,
                        "pattern": pattern,
                        "start": match.start(),
                        "matched_length": matched_length,
                        "family_index": family_index,
                        "pattern_index": pattern_index,
                    }
                )

    if matches:
        best_match = min(
            matches,
            key=lambda item: (
                item["start"],
                -item["matched_length"],
                item["family_index"],
                item["pattern_index"],
            ),
        )

        return (
            best_match["role_family"],
            "high",
            best_match["pattern"],
        )

    if fallback:
        return fallback, "low", "source_fallback"

    return "unknown", "unknown", "no_matching_rule"


def extract_experience_years(description, title):
    text = normalise_text(
        f"{title} {description}"
    )

    values = []

    range_patterns = [
        (
            r"(\d{1,2})\s*(?:-|–|—|to)\s*"
            r"(\d{1,2})\s*(?:years?|yrs?)"
            r"(?:\s+of)?\s+(?:relevant\s+)?experience"
        ),
        (
            r"(?:experience|experienced)\s*(?:of|:)?\s*"
            r"(\d{1,2})\s*(?:-|–|—|to)\s*"
            r"(\d{1,2})\s*(?:years?|yrs?)"
        ),
    ]

    for pattern in range_patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            lower_value = int(match.group(1))

            if 0 <= lower_value <= 30:
                values.append(lower_value)

    single_patterns = [
        (
            r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)"
            r"(?:\s+of)?\s+(?:relevant\s+)?experience"
        ),
        (
            r"(?:minimum|min\.?|at least)\s+"
            r"(\d{1,2})\s*(?:years?|yrs?)"
        ),
        (
            r"(?:experience|required experience)"
            r"\s*(?:of|:)?\s*(\d{1,2})"
            r"\s*\+?\s*(?:years?|yrs?)"
        ),
    ]

    for pattern in single_patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            value = int(match.group(1))

            if 0 <= value <= 30:
                values.append(value)

    if values:
        return max(values), "description"

    lowered_title = title.casefold()

    if any(
        term in lowered_title
        for term in [
            "intern",
            "trainee",
            "fresher",
            "graduate",
            "entry level",
        ]
    ):
        return 0, "title_inference"

    return None, "not_detected"


def derive_seniority(
    title,
    source_seniority,
    experience_years,
):
    lowered_title = normalise_text(
        title
    ).casefold()

    if re.search(
        r"\b(intern|internship|trainee|apprentice|fresher)\b",
        lowered_title,
    ):
        return "Intern", "title"

    if re.search(
        r"\b(chief|director|vice president|vp|head|"
        r"principal|staff)\b",
        lowered_title,
    ):
        return "Staff", "title"

    if re.search(
        r"\b(senior|sr\.?|lead|team leader|manager)\b",
        lowered_title,
    ):
        if re.search(
            r"\b(assistant|asst\.?|deputy|junior)\s+manager\b",
            lowered_title,
        ):
            return "Mid", "title"

        return "Senior", "title"

    if re.search(
        r"\b(junior|jr\.?|entry level|graduate|associate)\b",
        lowered_title,
    ):
        return "Junior", "title"

    source_key = normalise_text(
        source_seniority
    ).casefold()

    if source_key in SOURCE_SENIORITY_MAPPING:
        return (
            SOURCE_SENIORITY_MAPPING[source_key],
            "source_mapping",
        )

    if experience_years is not None:
        if experience_years >= 10:
            return "Staff", "experience"

        if experience_years >= 6:
            return "Senior", "experience"

        if experience_years >= 3:
            return "Mid", "experience"

        return "Junior", "experience"

    return "Unknown", "not_detected"


def normalise_candidate_seniority(value):
    mapping = {
        "junior": "Junior",
        "mid": "Mid",
        "senior": "Senior",
        "staff": "Staff",
        "intern": "Intern",
    }

    return mapping.get(
        normalise_text(value).casefold(),
        "Unknown",
    )


def normalise_location(location):
    location_text = normalise_text(
        location
    )

    lowered_location = location_text.casefold()

    remote = bool(
        re.search(
            r"\b(remote|work from home|wfh)\b",
            lowered_location,
        )
    )

    city = None

    # Longer names are checked first.
    for city_variant in sorted(
        INDIAN_CITY_NAMES,
        key=len,
        reverse=True,
    ):
        if re.search(
            r"\b"
            + re.escape(city_variant)
            + r"\b",
            lowered_location,
        ):
            city = INDIAN_CITY_NAMES[city_variant]
            break

    final_location_part = (
        location_text.split(",")[-1].strip()
        if "," in location_text
        else location_text.strip()
    )

    final_location_key = final_location_part.casefold()

    if (
        "india" in lowered_location
        or city is not None
    ):
        country = "India"

    elif (
        final_location_key in US_STATE_CODES
        or re.search(
            r"\b(united states|united states of america|usa|u\.s\.a\.)\b",
            lowered_location,
        )
        or "cincinnati metropolitan area" in lowered_location
    ):
        country = "United States"

    elif "united kingdom" in lowered_location:
        country = "United Kingdom"

    elif "greater rio de janeiro" in lowered_location:
        country = "Brazil"

    elif remote and not location_text:
        country = "Unknown"

    elif "," in location_text:
        country = final_location_part

    elif location_text:
        country = location_text

    else:
        country = "Unknown"

    return {
        "original": location_text,
        "city": city,
        "country": country,
        "remote_in_location": remote,
    }


def normalise_employment_type(value):
    lowered = normalise_text(value).casefold()

    mapping = {
        "full-time": "Full-time",
        "full time": "Full-time",
        "part-time": "Part-time",
        "part time": "Part-time",
        "contract": "Contract",
        "temporary": "Temporary",
        "internship": "Internship",
        "volunteer": "Volunteer",
    }

    return mapping.get(
        lowered,
        normalise_text(value) or "Unknown",
    )


def main():
    jobs = load_json("jobs_skills_extracted.json")
    profiles = load_json(
        "profiles_skills_normalized.json"
    )

    enriched_jobs = []
    role_disagreements = []
    job_role_counter = Counter()
    seniority_counter = Counter()
    experience_source_counter = Counter()
    country_counter = Counter()
    remote_counter = Counter()

    for job in jobs:
        source_role = job.get(
            "role_family_hint",
            "unknown",
        )

        (
            derived_role,
            role_confidence,
            role_rule,
        ) = derive_role_family(
            job.get("title", ""),
            fallback=source_role,
        )

        if (
            role_confidence == "high"
            and derived_role != source_role
        ):
            role_disagreements.append(
                {
                    "id": job["id"],
                    "title": job.get("title"),
                    "source_role": source_role,
                    "derived_role": derived_role,
                    "matched_rule": role_rule,
                }
            )

        experience_years, experience_source = (
            extract_experience_years(
                job.get("description", ""),
                job.get("title", ""),
            )
        )

        seniority, seniority_source = (
            derive_seniority(
                job.get("title", ""),
                job.get("seniority_level", ""),
                experience_years,
            )
        )

        location_features = normalise_location(
            job.get("location", "")
        )

        remote_available = bool(
            job.get("remote_hint", False)
            or location_features["remote_in_location"]
            or re.search(
                r"\bremote\b",
                normalise_text(
                    job.get("title", "")
                ),
                flags=re.IGNORECASE,
            )
        )

        domain = (
            normalise_text(job.get("industry"))
            or "Unknown"
        )

        enriched_job = dict(job)

        enriched_job.update(
            {
                "source_role_family": source_role,
                "derived_role_family": derived_role,
                "role_derivation_confidence":
                    role_confidence,
                "role_derivation_rule": role_rule,
                "minimum_experience_years":
                    experience_years,
                "experience_extraction_source":
                    experience_source,
                "seniority_normalized": seniority,
                "seniority_ordinal":
                    SENIORITY_ORDINAL[seniority],
                "seniority_derivation_source":
                    seniority_source,
                "location_normalized":
                    location_features,
                "remote_available":
                    remote_available,
                "employment_type_normalized":
                    normalise_employment_type(
                        job.get("employment_type")
                    ),
                "domain_normalized": domain,
            }
        )

        enriched_jobs.append(enriched_job)

        job_role_counter[derived_role] += 1
        seniority_counter[seniority] += 1
        experience_source_counter[
            experience_source
        ] += 1
        country_counter[
            location_features["country"]
        ] += 1
        remote_counter[
            "remote" if remote_available else "not_remote"
        ] += 1

    enriched_profiles = []
    candidate_role_counter = Counter()
    unknown_candidate_role_titles = Counter()

    for profile in profiles:
        candidate_roles = []
        role_details = []

        for role_title in profile.get("roles", []):
            role_family, confidence, rule = (
                derive_role_family(role_title)
            )

            role_details.append(
                {
                    "role_title": role_title,
                    "role_family": role_family,
                    "confidence": confidence,
                    "matched_rule": rule,
                }
            )

            if role_family != "unknown":
                candidate_roles.append(role_family)
            else:
                unknown_candidate_role_titles[
                    role_title
                ] += 1

        candidate_roles = list(
            dict.fromkeys(candidate_roles)
        )

        if not candidate_roles:
            candidate_roles = ["unknown"]

        seniority = normalise_candidate_seniority(
            profile.get("seniority")
        )

        preference_locations = [
            normalise_location(location)
            for location in (
                profile
                .get("preferences", {})
                .get("locations", [])
            )
        ]

        remote_preference = bool(
            profile
            .get("preferences", {})
            .get("remote", False)
        )

        enriched_profile = dict(profile)

        enriched_profile.update(
            {
                "derived_role_families":
                    candidate_roles,
                "role_derivation_details":
                    role_details,
                "seniority_normalized":
                    seniority,
                "seniority_ordinal":
                    SENIORITY_ORDINAL[seniority],
                "preference_locations_normalized":
                    preference_locations,
                "remote_preference":
                    remote_preference,
            }
        )

        enriched_profiles.append(enriched_profile)

        for role_family in candidate_roles:
            candidate_role_counter[role_family] += 1

    report = {
        "input": {
            "job_count": len(jobs),
            "profile_count": len(profiles),
        },
        "output": {
            "enriched_job_count":
                len(enriched_jobs),
            "enriched_profile_count":
                len(enriched_profiles),
            "high_confidence_role_disagreements":
                len(role_disagreements),
            "jobs_with_experience_detected":
                sum(
                    job["minimum_experience_years"]
                    is not None
                    for job in enriched_jobs
                ),
            "jobs_without_experience_detected":
                sum(
                    job["minimum_experience_years"]
                    is None
                    for job in enriched_jobs
                ),
        },
        "job_role_distribution":
            dict(job_role_counter.most_common()),
        "candidate_role_distribution":
            dict(candidate_role_counter.most_common()),
            "unknown_candidate_role_titles": [
            {
                "role_title": role_title,
                "profile_count": count,
            }
            for role_title, count
            in unknown_candidate_role_titles.most_common()],
        "job_seniority_distribution":
            dict(seniority_counter.most_common()),
        "experience_extraction_sources":
            dict(experience_source_counter.most_common()),
        "country_distribution_top_25":
            dict(country_counter.most_common(25)),
        "remote_distribution":
            dict(remote_counter),
        "role_disagreement_sample":
            role_disagreements[:100],
        "safety_rules": {
            "source_role_preserved": True,
            "source_role_blindly_overwritten": False,
            "derived_role_confidence_recorded": True,
            "experience_missing_values_invented":
                False,
            "generated_labels_used": False,
            "job_skills_preserved": True,
        },
    }

    save_json(
        "jobs_features_enriched.json",
        enriched_jobs,
    )

    save_json(
        "profiles_features_enriched.json",
        enriched_profiles,
    )

    save_json(
        "feature_preparation_report.json",
        report,
    )

    print("Stage 4B feature preparation completed.")
    print("")
    print(f"Jobs processed: {len(enriched_jobs):,}")
    print(
        f"Profiles processed: "
        f"{len(enriched_profiles):,}"
    )
    print(
        "Jobs with experience detected: "
        f"{report['output']['jobs_with_experience_detected']:,}"
    )
    print(
        "Role disagreements requiring review: "
        f"{len(role_disagreements):,}"
    )
    print("")
    print(
        "Report created at: "
        f"{PROCESSED_DIR / 'feature_preparation_report.json'}"
    )


if __name__ == "__main__":
    main()