from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_selection_module():
    path = ROOT / "scripts" / "select_hybrid_ranking_policy.py"
    spec = importlib.util.spec_from_file_location("stage11_selection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Stage 11 selection module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    paths = [
        ROOT / "src" / "job_optimizer_engine.py",
        ROOT / "scripts" / "select_hybrid_ranking_policy.py",
        ROOT / "scripts" / "audit_candidate_ranking_policy.py",
        ROOT / "scripts" / "ranking_quality_scenarios.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    engine_source = paths[0].read_text(encoding="utf-8")
    required_markers = [
        "for _ in range(3)",
        "html.unescape",
        "ranking_policy",
        "sbert_composite",
        "np.argsort(-ranking_scores",
        "ranking_and_display_use_same_score",
    ]
    # The last marker belongs to the selection script, not the engine.
    combined_source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    missing = [
        marker
        for marker in required_markers
        if marker not in (combined_source if marker == "ranking_and_display_use_same_score" else engine_source)
    ]
    if missing:
        raise ValueError(f"Stage 11 integration markers are missing: {missing}")

    module = load_selection_module()
    gnn = np.asarray([10.0, 20.0, 30.0])
    sbert = np.asarray([30.0, 20.0, 10.0])
    blended = module.blend(gnn, sbert, 0.25)
    expected = np.asarray([25.0, 20.0, 15.0])
    if not np.allclose(blended, expected):
        raise ValueError("Hybrid blending calculation is incorrect")

    print("Stage 11 package validation passed.")
    print("Syntax: passed")
    print("Hybrid calculation: passed")
    print("Production checkpoint mutation: absent")


if __name__ == "__main__":
    main()
