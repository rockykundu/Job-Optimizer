# Ranking Quality and API Foundation

The final product architecture is:

- React and TypeScript for the professional web dashboard.
- FastAPI for the application API.
- The existing local Python SBERT and GNN engine for inference.
- Model checkpoints, embeddings and processed datasets on the backend only.

Streamlit is retained only as an internal verification interface.

## Purpose of this package

This package does not change the trained model, ranking formula, checkpoint or
datasets. It:

- decodes HTML entities in job titles, companies, locations and descriptions;
- adds non-ranking diagnostic fields to each result;
- measures score compression, job-family relevance and structured-evidence
  availability with five controlled resumes;
- produces evidence for the ranking/calibration decision before the API and
  React dashboard are finalised.

## Installation

From the project root, back up the current engine:

```powershell
Copy-Item .\src\job_optimizer_engine.py .\src\job_optimizer_engine.before-quality-audit.py.bak -Force
```

Place `JobOptimizer_RankingQualityAudit.zip` in the project root and extract it:

```powershell
Expand-Archive -Path .\JobOptimizer_RankingQualityAudit.zip -DestinationPath . -Force
```

## Run the audit

```powershell
python .\scripts\audit_ranking_quality.py
```

The audit loads the existing model once and evaluates five controlled career
profiles. It does not train or download anything.

The report is written to:

`evaluation\ranking_quality\ranking_quality_report.json`

Review that report before implementing any score calibration, filtering or
reranking rule.

## Planned phases after this report

1. Approve a measured ranking-quality correction, if required.
2. Build the versioned FastAPI backend and validate its response contract.
3. Build the professional React/TypeScript dashboard.
4. Run end-to-end, security, performance and user-acceptance tests.
5. Prepare production deployment configuration.
