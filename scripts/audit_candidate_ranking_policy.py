from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ranking_quality_scenarios import SCENARIOS
from src.job_optimizer_engine import JobOptimizerEngine


POLICY_PATH = ROOT / "models" / "candidate" / "hybrid_ranking_policy.json"
OUTPUT_PATH = (
    ROOT
    / "evaluation"
    / "ranking_policy"
    / "controlled_profile_comparison.json"
)
ENTITY_PATTERN = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.IGNORECASE)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def summarise(result: dict[str, Any], expected_pattern: str) -> dict[str, Any]:
    matches = result["matches"]
    expected = re.compile(expected_pattern, re.IGNORECASE)
    relevant_ranks = [
        int(match["rank"]) for match in matches if expected.search(match["title"])
    ]
    scores = [float(match["ranking_score"]) for match in matches[:10]]
    entity_failures = [
        match["title"] for match in matches if ENTITY_PATTERN.search(match["title"])
    ]
    return {
        "ranking_policy": result["ranking_policy"],
        "first_expected_job_family_rank": relevant_ranks[0] if relevant_ranks else None,
        "top_five_expected_job_family_hits": sum(
            bool(expected.search(match["title"])) for match in matches[:5]
        ),
        "top_ten_expected_job_family_hits": sum(
            bool(expected.search(match["title"])) for match in matches[:10]
        ),
        "top_ten_score_spread": round(max(scores) - min(scores), 6),
        "structured_evidence_top_twenty": sum(
            bool(match["structured_skill_evidence_available"]) for match in matches
        ),
        "unresolved_html_entity_titles": entity_failures,
        "top_results": [
            {
                "rank": match["rank"],
                "job_id": match["job_id"],
                "title": match["title"],
                "company": match["company"],
                "ranking_score": round(float(match["ranking_score"]), 6),
                "gnn_composite_score": round(
                    float(match["raw_composite_score"]), 6
                ),
                "sbert_calibrated_composite_score": (
                    None
                    if match["sbert_calibrated_composite_score"] is None
                    else round(
                        float(match["sbert_calibrated_composite_score"]), 6
                    )
                ),
                "cosine_similarity": round(
                    float(match["job_feature_cosine_similarity"]), 6
                ),
                "matched_structured_skills": match["unique_matched_skill_count"],
                "available_structured_skills": match[
                    "unique_structured_skill_count"
                ],
            }
            for match in matches[:10]
        ],
    }


def main() -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {POLICY_PATH}. Run select_hybrid_ranking_policy.py first."
        )
    policy = load_json(POLICY_PATH)
    engine = JobOptimizerEngine(ROOT)

    comparisons = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"[{index}/{len(SCENARIOS)}] Comparing {scenario['id']}...")
        gnn_result = engine.rank_resume(scenario["resume"], top_k=20)
        hybrid_result = engine.rank_resume(
            scenario["resume"], top_k=20, ranking_policy=policy
        )
        gnn = summarise(gnn_result, scenario["expected_title_pattern"])
        hybrid = summarise(hybrid_result, scenario["expected_title_pattern"])
        comparisons.append(
            {
                "scenario_id": scenario["id"],
                "gnn_only": gnn,
                "candidate_hybrid": hybrid,
                "top_five_expected_hit_change": (
                    hybrid["top_five_expected_job_family_hits"]
                    - gnn["top_five_expected_job_family_hits"]
                ),
                "top_ten_expected_hit_change": (
                    hybrid["top_ten_expected_job_family_hits"]
                    - gnn["top_ten_expected_job_family_hits"]
                ),
                "score_spread_change": round(
                    hybrid["top_ten_score_spread"] - gnn["top_ten_score_spread"],
                    6,
                ),
            }
        )

    gnn_top_five = sum(
        item["gnn_only"]["top_five_expected_job_family_hits"]
        for item in comparisons
    )
    hybrid_top_five = sum(
        item["candidate_hybrid"]["top_five_expected_job_family_hits"]
        for item in comparisons
    )
    gnn_top_ten = sum(
        item["gnn_only"]["top_ten_expected_job_family_hits"]
        for item in comparisons
    )
    hybrid_top_ten = sum(
        item["candidate_hybrid"]["top_ten_expected_job_family_hits"]
        for item in comparisons
    )
    entity_failures = sum(
        len(item["candidate_hybrid"]["unresolved_html_entity_titles"])
        for item in comparisons
    )
    report = {
        "stage": "Stage 11 controlled-profile candidate policy comparison",
        "status": "completed",
        "promotion_status": "manual_review_required",
        "candidate_policy": policy,
        "aggregate": {
            "gnn_top_five_expected_hits_out_of_25": gnn_top_five,
            "hybrid_top_five_expected_hits_out_of_25": hybrid_top_five,
            "top_five_expected_hit_change": hybrid_top_five - gnn_top_five,
            "gnn_top_ten_expected_hits_out_of_50": gnn_top_ten,
            "hybrid_top_ten_expected_hits_out_of_50": hybrid_top_ten,
            "top_ten_expected_hit_change": hybrid_top_ten - gnn_top_ten,
            "candidate_hybrid_unresolved_html_entity_count": entity_failures,
        },
        "comparisons": comparisons,
        "safety": {
            "candidate_policy_not_promoted": True,
            "production_checkpoint_unchanged": True,
            "ranking_comparison_does_not_train": True,
            "dataset_download_requested": False,
            "model_download_requested": False,
        },
    }
    save_json(OUTPUT_PATH, report)

    print("Controlled-profile comparison completed.")
    print(f"GNN top-five expected-family hits: {gnn_top_five}/25")
    print(f"Hybrid top-five expected-family hits: {hybrid_top_five}/25")
    print(f"Remaining HTML entity titles: {entity_failures}")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
