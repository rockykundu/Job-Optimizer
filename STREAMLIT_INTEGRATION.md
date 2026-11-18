# Streamlit Production Integration

This package replaces the prototype Streamlit scoring path with the validated
local production inference engine.

## What this phase changes

- Replaces the prototype `app.py`.
- Adds `src/job_optimizer_engine.py`, the production inference interface.
- Adds `scripts/validate_streamlit_integration.py`.
- Updates `requirements.txt` with the runtime dependencies used by the engine.

It does not replace, retrain, or download datasets, embeddings, or model
checkpoints.

## Installation

Run these commands separately from the project root in PowerShell:

```powershell
Copy-Item .\app.py .\app.prototype.backup.py -Force
```

```powershell
Copy-Item .\requirements.txt .\requirements.prototype.backup.txt -Force
```

Place `JobOptimizer_StreamlitIntegration.zip` in the project root, then run:

```powershell
Expand-Archive -Path .\JobOptimizer_StreamlitIntegration.zip -DestinationPath . -Force
```

## Validation

```powershell
python .\scripts\validate_streamlit_integration.py
```

The validation report is written to:

`evaluation\streamlit_integration\streamlit_integration_report.json`

Start the application only after validation passes:

```powershell
streamlit run app.py
```

## Runtime behavior

- Resume PDFs are processed locally.
- Existing local SBERT assets and the saved GNN checkpoint are reused.
- Jobs are ranked by raw model output; clipping is display-only.
- Skill overlap is explanatory and does not replace or modify the numeric model score.
- Jobs without structured skill evidence are labelled as unavailable rather than
  being shown as having no skill gaps.
