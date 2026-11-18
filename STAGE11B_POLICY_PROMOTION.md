# Stage 11B — Production Ranking Policy Promotion

This package promotes the approved validation-selected ranking policy:

- GNN: 0.78
- calibrated SBERT: 0.22
- production version: 1.1.0

It does not retrain the model, replace the GNN checkpoint, download data or
download model assets.

## Safety behaviour

The promotion script refuses to proceed unless:

- the candidate policy contains the approved weights;
- validation guardrails passed;
- the selection and candidate policy agree;
- controlled top-five relevance did not regress;
- controlled top-ten relevance improved;
- no unresolved HTML-title entities remain;
- the checkpoint-safety assertion is present.

Any existing production policy and model configuration are copied into
`models/final/backups` before version 1.1 is written.

## Run from PowerShell

Back up the current engine:

```powershell
Copy-Item .\src\job_optimizer_engine.py .\src\job_optimizer_engine.before-promotion.py.bak -Force
```

Extract the package:

```powershell
Expand-Archive -Path .\JobOptimizer_Stage11B_PolicyPromotion.zip -DestinationPath . -Force
```

Validate the package:

```powershell
python .\scripts\validate_policy_promotion_package.py
```

Promote the candidate policy:

```powershell
python .\scripts\promote_hybrid_ranking_policy.py
```

Run the production runtime validation:

```powershell
python .\scripts\validate_production_ranking_policy.py
```

The runtime validation executes the five controlled profiles and must reproduce
all 50 approved top-ten job IDs and scores.

## Reports

- `evaluation\ranking_policy\production_policy_promotion.json`
- `evaluation\ranking_policy\production_policy_validation_report.json`

Do not launch the application or start FastAPI if runtime validation fails.
Keep the terminal output and send it for diagnosis; the recorded backups provide
the rollback source.

## Next phase

After the runtime report passes, build the versioned FastAPI backend that will
serve the professional React/TypeScript dashboard.
