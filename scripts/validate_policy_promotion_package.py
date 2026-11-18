from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = [
        ROOT / "src" / "job_optimizer_engine.py",
        ROOT / "scripts" / "promote_hybrid_ranking_policy.py",
        ROOT / "scripts" / "validate_production_ranking_policy.py",
        ROOT / "scripts" / "ranking_quality_scenarios.py",
    ]
    combined = ""
    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        combined += "\n" + source

    required = [
        "candidate_not_promoted",
        "accepted_by_validation_guardrails",
        "backup_if_present",
        "promoted_pending_runtime_validation",
        "use_production_policy",
        "top_ten_job_ids_exact",
        "ranking_and_display_use_same_hybrid_score",
    ]
    missing = [marker for marker in required if marker not in combined]
    if missing:
        raise ValueError(f"Promotion safeguards are missing: {missing}")
    forbidden = ["torch.save(", ".backward(", "optimizer.step(", "requests.get("]
    present = [marker for marker in forbidden if marker in combined]
    if present:
        raise ValueError(f"Unexpected training or network mutation code: {present}")

    print("Policy-promotion package validation passed.")
    print("Syntax: passed")
    print("Evidence guardrails: present")
    print("Rollback backup logic: present")
    print("Training and download logic: absent")


if __name__ == "__main__":
    main()
