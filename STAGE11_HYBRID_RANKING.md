# Stage 11 — Validation-Selected Hybrid Ranking

This stage evaluates whether a GNN/SBERT hybrid produces better candidate-centred
rankings than the current GNN-only policy.

It does not retrain or replace the GNN checkpoint. It creates a candidate policy
under `models/candidate`; the production policy is not promoted automatically.

## Selection protocol

1. Load only validation GNN and SBERT predictions.
2. Search convex GNN/SBERT weights in increments of 0.01.
3. Optimise candidate-centred composite NDCG@10.
4. Use pairwise ranking accuracy as the secondary objective.
5. Reject a candidate that violates the MAE or pairwise guardrails.
6. Freeze the selected weights.
7. Load test and gold predictions only after freezing the policy.
8. Compare GNN-only and candidate-hybrid rankings on five controlled profiles.

## Install

From the project root:

```powershell
Copy-Item .\src\job_optimizer_engine.py .\src\job_optimizer_engine.before-stage11.py.bak -Force
```

```powershell
Expand-Archive -Path .\JobOptimizer_Stage11_HybridRanking.zip -DestinationPath . -Force
```

## Validate code

```powershell
python .\scripts\validate_stage11_package.py
```

## Select candidate policy

```powershell
python .\scripts\select_hybrid_ranking_policy.py
```

This creates:

- `models\candidate\hybrid_ranking_policy.json`
- `evaluation\ranking_policy\hybrid_ranking_selection_report.json`

## Run controlled comparison

```powershell
python .\scripts\audit_candidate_ranking_policy.py
```

This creates:

`evaluation\ranking_policy\controlled_profile_comparison.json`

Both reports must be reviewed before the candidate policy can be promoted.

## Architecture boundary

The eventual React frontend will receive only versioned API responses. It will
not load model files or calculate rankings in the browser. The FastAPI service
will own PDF processing, inference, ranking policy application and response
validation.
