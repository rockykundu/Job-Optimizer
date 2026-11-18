from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.job_optimizer_engine import JobOptimizerEngine


OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "ranking_quality"
    / "ranking_quality_report.json"
)
ENTITY_PATTERN = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.IGNORECASE)

SCENARIOS = [
    {
        "id": "backend_engineering",
        "expected_title_pattern": r"backend|back-end|software engineer|python|django|api|platform engineer",
        "resume": """
        Backend software engineer with five years of experience building Python,
        Django and FastAPI services. Strong PostgreSQL, Redis, REST API, Docker,
        AWS, Linux, Git, automated testing and CI/CD experience. Designed reliable
        distributed services and production monitoring.
        """,
    },
    {
        "id": "frontend_engineering",
        "expected_title_pattern": r"frontend|front-end|react|javascript|typescript|web developer|ui developer",
        "resume": """
        Frontend engineer specialising in React, TypeScript, JavaScript, HTML,
        CSS, Next.js, accessible design systems, responsive web applications,
        Git, REST APIs, Jest and frontend performance optimisation.
        """,
    },
    {
        "id": "mobile_engineering",
        "expected_title_pattern": r"mobile|android|ios|flutter|react native",
        "resume": """
        Mobile application developer experienced with Flutter, Dart, Android,
        iOS, Firebase, REST APIs, Git, unit testing and publishing production
        applications to mobile app stores.
        """,
    },
    {
        "id": "data_and_ai",
        "expected_title_pattern": r"data|machine learning|artificial intelligence|\bai\b|analytics|scientist",
        "resume": """
        Data scientist and machine learning engineer using Python, pandas,
        scikit-learn, PyTorch, SQL, statistics, data analysis, deep learning,
        natural language processing, Tableau, experiment design and model
        deployment.
        """,
    },
    {
        "id": "devops_and_cloud",
        "expected_title_pattern": r"devops|site reliability|\bsre\b|cloud engineer|platform engineer|infrastructure",
        "resume": """
        DevOps and cloud engineer with Kubernetes, Docker, Terraform, AWS,
        Linux, networking, GitHub Actions, CI/CD, infrastructure as code,
        observability, incident response and production reliability experience.
        """,
    },
]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def round_optional(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def analyse_scenario(engine: JobOptimizerEngine, scenario: dict[str, str]) -> dict[str, Any]:
    result = engine.rank_resume(scenario["resume"], top_k=20)
    matches = result["matches"]
    expected = re.compile(scenario["expected_title_pattern"], re.IGNORECASE)
    relevant_ranks = [
        int(match["rank"]) for match in matches if expected.search(match["title"])
    ]
    composite_scores = [float(match["raw_composite_score"]) for match in matches]
    skill_scores = [float(match["raw_skills_score"]) for match in matches]
    top_ten = composite_scores[:10]
    top_five = matches[:5]
    structured_count = sum(
        bool(match["structured_skill_evidence_available"]) for match in matches
    )
    unresolved_entities = [
        {
            "rank": match["rank"],
            "job_id": match["job_id"],
            "title": match["title"],
        }
        for match in matches
        if ENTITY_PATTERN.search(match["title"])
    ]
    first_expected_rank = relevant_ranks[0] if relevant_ranks else None
    top_five_expected_hits = sum(bool(expected.search(match["title"])) for match in top_five)
    score_spread = max(top_ten) - min(top_ten)
    findings: list[str] = []
    if first_expected_rank is None:
        findings.append("No expected job-family title appeared in the top 20.")
    elif first_expected_rank > 5:
        findings.append(
            f"The first expected job-family title appeared at rank {first_expected_rank}."
        )
    if top_five_expected_hits < 3:
        findings.append(
            f"Only {top_five_expected_hits}/5 top results matched the expected job family."
        )
    if score_spread < 2.0:
        findings.append(
            f"The top-10 composite-score spread was compressed to {score_spread:.4f} points."
        )
    if structured_count < 10:
        findings.append(
            f"Only {structured_count}/20 results had structured skill evidence."
        )
    if unresolved_entities:
        findings.append("One or more displayed titles still contained HTML entities.")

    return {
        "scenario_id": scenario["id"],
        "expected_title_pattern": scenario["expected_title_pattern"],
        "detected_skills": result["candidate"]["detected_skills"],
        "first_expected_job_family_rank": first_expected_rank,
        "top_five_expected_job_family_hits": top_five_expected_hits,
        "top_ten_composite_score_spread": round(score_spread, 6),
        "top_ten_composite_score_standard_deviation": round(
            statistics.pstdev(top_ten), 6
        ),
        "top_twenty_skill_score_spread": round(max(skill_scores) - min(skill_scores), 6),
        "structured_evidence_result_count": structured_count,
        "unresolved_html_entities": unresolved_entities,
        "findings": findings,
        "top_results": [
            {
                "rank": match["rank"],
                "job_id": match["job_id"],
                "title": match["title"],
                "company": match["company"],
                "raw_composite_score": round(
                    float(match["raw_composite_score"]), 6
                ),
                "raw_skills_score": round(float(match["raw_skills_score"]), 6),
                "job_feature_cosine_similarity": round(
                    float(match["job_feature_cosine_similarity"]), 6
                ),
                "raw_score_gap_from_top": round(
                    float(match["raw_score_gap_from_top"]), 6
                ),
                "unique_structured_skill_count": match[
                    "unique_structured_skill_count"
                ],
                "unique_matched_skill_count": match["unique_matched_skill_count"],
                "structured_skill_overlap_ratio": round_optional(
                    match["structured_skill_overlap_ratio"]
                ),
            }
            for match in matches
        ],
    }


def main() -> None:
    engine = JobOptimizerEngine(PROJECT_ROOT)
    scenario_reports = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"[{index}/{len(SCENARIOS)}] Auditing {scenario['id']}...")
        scenario_reports.append(analyse_scenario(engine, scenario))

    finding_count = sum(len(item["findings"]) for item in scenario_reports)
    report = {
        "stage": "Ranking quality and calibration audit",
        "status": "completed_with_findings" if finding_count else "completed",
        "quality_gate": "manual_review_required",
        "rankable_job_count": len(engine.main_job_ids),
        "scenario_count": len(scenario_reports),
        "finding_count": finding_count,
        "model_behaviour": {
            "ranking_formula_changed": False,
            "checkpoint_changed": False,
            "training_run": False,
            "dataset_download_requested": False,
            "model_download_requested": False,
            "display_html_entities_decoded": True,
            "diagnostic_fields_do_not_change_ranking": True,
        },
        "scenarios": scenario_reports,
    }
    save_json(OUTPUT_PATH, report)

    print("Ranking quality audit completed.")
    print(f"Scenarios: {len(scenario_reports)}")
    print(f"Findings requiring review: {finding_count}")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
