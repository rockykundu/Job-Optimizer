# Job Optimizer React Frontend — Stage 13

This folder is the professional React + TypeScript interface for the validated Stage 12 FastAPI backend.

## What is included

- Drag-and-drop PDF resume upload with 10 MB client-side validation
- Real `/api/v1/model` and `/api/v1/analyze-resume` integration
- Ranked job cards with role, company, location, score, and skill evidence
- Detailed compatibility panel with matched and missing skills
- Evidence-confidence messaging for incomplete job records
- Search across returned role, company, and location fields
- Loading, empty, invalid-file, API-error, and success states
- Responsive desktop, tablet, and mobile layouts
- Accessible keyboard focus, labels, and reduced-motion support
- No mock rankings, resume persistence, training, or dataset downloads

## Folder location

Extract the ZIP into:

```text
C:\Users\rocky\Documents\websiter.bh\Job-Optimizer
```

The result should be:

```text
Job-Optimizer
├── api
├── data
├── models
├── scripts
├── src
└── frontend
```

## First-time setup

Open a new PowerShell terminal and run each command separately:

```powershell
cd C:\Users\rocky\Documents\websiter.bh\Job-Optimizer\frontend
```

```powershell
Copy-Item .env.example .env
```

```powershell
npm install
```

```powershell
npm run validate
```

```powershell
npm run build
```

## Start the application

### Terminal 1 — backend

```powershell
cd C:\Users\rocky\Documents\websiter.bh\Job-Optimizer
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Leave this terminal running.

### Terminal 2 — frontend

```powershell
cd C:\Users\rocky\Documents\websiter.bh\Job-Optimizer\frontend
```

```powershell
npm run dev
```

Open `http://127.0.0.1:5173` in your browser.

## Validation output

After `npm run validate`, upload this file for review:

```text
C:\Users\rocky\Documents\websiter.bh\Job-Optimizer\frontend\frontend_validation.json
```

Also send a screenshot after testing one real PDF resume.

## Important

- Keep the backend and frontend terminals open while using the app.
- Do not put secrets in variables beginning with `VITE_`; browser users can read them.
- The compatibility score is a model estimate out of 100, not a hiring probability.
- `npm install` is required only initially or when `package.json` changes. It does not download the training dataset.
