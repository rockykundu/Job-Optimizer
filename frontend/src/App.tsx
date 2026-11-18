import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  FileSearch,
  History,
  LayoutDashboard,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
} from 'lucide-react'
import { JobCard } from './components/JobCard'
import { JobDetail } from './components/JobDetail'
import { UploadPanel } from './components/UploadPanel'
import { analyzeResume, ApiError, getModelInfo } from './lib/api'
import type { JobMatch, ModelInfo, ResumeAnalysisResponse } from './types'

type AnalysisState = 'idle' | 'loading' | 'success' | 'error'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [topK, setTopK] = useState(10)
  const [state, setState] = useState<AnalysisState>('idle')
  const [result, setResult] = useState<ResumeAnalysisResponse | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [modelOnline, setModelOnline] = useState<boolean | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [query, setQuery] = useState('')
  const [mobileDetail, setMobileDetail] = useState(false)
  const requestController = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    getModelInfo(controller.signal)
      .then((data) => { setModel(data); setModelOnline(true) })
      .catch(() => setModelOnline(false))
    return () => controller.abort()
  }, [])

  useEffect(() => () => requestController.current?.abort(), [])

  const matches = useMemo(() => {
    if (!result) return []
    const normalized = query.trim().toLowerCase()
    if (!normalized) return result.matches
    return result.matches.filter((match) =>
      [match.title, match.company, match.location].some((value) => value.toLowerCase().includes(normalized)),
    )
  }, [query, result])

  const selected = result?.matches.find((match) => match.job_id === selectedId) ?? result?.matches[0] ?? null

  const chooseFile = (nextFile: File | null) => {
    setError(null)
    if (!nextFile) { setFile(null); return }
    if (!nextFile.name.toLowerCase().endsWith('.pdf') || !['application/pdf', 'application/x-pdf', ''].includes(nextFile.type)) {
      setError(new ApiError('Select a PDF resume. Other document types are not supported.'))
      setFile(null)
      return
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError(new ApiError('The selected PDF is larger than 10 MB.'))
      setFile(null)
      return
    }
    setFile(nextFile)
  }

  const runAnalysis = async () => {
    if (!file) return
    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller
    setState('loading')
    setError(null)
    setQuery('')
    try {
      const response = await analyzeResume(file, topK, controller.signal)
      setResult(response)
      setSelectedId(response.matches[0]?.job_id ?? null)
      setState('success')
    } catch (caught) {
      if (controller.signal.aborted) return
      setError(caught instanceof ApiError ? caught : new ApiError('The analysis could not be completed.'))
      setState('error')
    }
  }

  const selectMatch = (match: JobMatch) => {
    setSelectedId(match.job_id)
    setMobileDetail(true)
  }

  const newAnalysis = () => {
    requestController.current?.abort()
    setResult(null)
    setSelectedId(null)
    setQuery('')
    setState('idle')
    setError(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand__mark"><Sparkles size={20} /></span><span>Job Optimizer</span></div>
        <nav aria-label="Primary navigation">
          <a className="nav-item is-active" href="#workspace"><LayoutDashboard size={19} /><span>Workspace</span></a>
          <span className="nav-item is-disabled"><History size={19} /><span>History</span><small>Soon</small></span>
          <span className="nav-item is-disabled"><BarChart3 size={19} /><span>Insights</span><small>Soon</small></span>
        </nav>
        <div className="sidebar__footer">
          <div className="system-state">
            <span className={`status-indicator ${modelOnline === false ? 'is-offline' : ''}`} />
            <div><strong>{modelOnline === false ? 'Service offline' : modelOnline ? 'Model ready' : 'Checking service'}</strong><small>{model ? `v${model.model_version}` : 'FastAPI connection'}</small></div>
          </div>
          <span className="nav-item is-disabled"><Settings size={19} /><span>Settings</span></span>
        </div>
      </aside>

      <main id="workspace" className="workspace">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand__mark"><Sparkles size={18} /></span><strong>Job Optimizer</strong></div>
          <div className="topbar__context"><span>Candidate workspace</span><strong>Resume matching</strong></div>
          {result && <button className="text-button" type="button" onClick={newAnalysis}><RefreshCw size={16} /> New analysis</button>}
        </header>

        <div className="workspace__content">
          {!result && (
            <UploadPanel file={file} topK={topK} busy={state === 'loading'} onFile={chooseFile} onTopK={setTopK} onAnalyze={runAnalysis} />
          )}

          {error && (
            <div className="error-banner" role="alert">
              <AlertTriangle size={20} />
              <div><strong>Analysis unavailable</strong><span>{error.message}</span>{error.requestId && <small>Request ID: {error.requestId}</small>}</div>
              {file && <button type="button" onClick={runAnalysis}>Try again</button>}
            </div>
          )}

          {state === 'loading' && (
            <section className="analysis-loading" aria-live="polite">
              <div className="analysis-loading__visual"><FileSearch size={28} /><span className="scan-line" /></div>
              <div><strong>Evaluating your resume</strong><span>Detecting skills and ranking {model?.rankable_job_count?.toLocaleString() ?? 'available'} roles…</span></div>
            </section>
          )}

          {result && (
            <section className="results" aria-labelledby="results-title">
              <div className="results-summary">
                <div className="results-summary__title">
                  <span className="success-icon"><CheckCircle2 size={18} /></span>
                  <div><span>Analysis complete</span><h1 id="results-title">Your strongest job matches</h1></div>
                </div>
                <div className="summary-metrics">
                  <div><span>Skills detected</span><strong>{result.candidate.detected_skill_count}</strong></div>
                  <div><span>Jobs evaluated</span><strong>{model?.rankable_job_count?.toLocaleString() ?? '2,468'}</strong></div>
                  <div><span>Analysis time</span><strong>{(result.processing_time_ms / 1000).toFixed(1)}s</strong></div>
                </div>
              </div>

              <div className="detected-skills">
                <span>Detected profile</span>
                <div>{result.candidate.detected_skills.slice(0, 10).map((skill) => <span key={skill}>{skill}</span>)}{result.candidate.detected_skill_count > 10 && <span>+{result.candidate.detected_skill_count - 10} more</span>}</div>
              </div>

              <div className="results-toolbar">
                <div className="search-field"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter by role, company, or location" aria-label="Filter job matches" /></div>
                <span>{matches.length} of {result.matches.length} matches</span>
              </div>

              <div className="results-grid">
                <div className="job-list" aria-label="Ranked job matches">
                  {matches.map((match) => <JobCard key={match.job_id} match={match} selected={selected?.job_id === match.job_id} onSelect={() => selectMatch(match)} />)}
                  {matches.length === 0 && <div className="no-results"><Search size={24} /><strong>No matching results</strong><span>Clear the filter to see all ranked jobs.</span></div>}
                </div>
                {selected && <JobDetail match={selected} mobileOpen={mobileDetail} onClose={() => setMobileDetail(false)} />}
              </div>

              <footer className="disclaimer"><AlertTriangle size={15} /><span>{result.disclaimer}</span><small>Request {result.request_id}</small></footer>
            </section>
          )}
        </div>
      </main>
      {mobileDetail && <button className="detail-backdrop" type="button" aria-label="Close details" onClick={() => setMobileDetail(false)} />}
    </div>
  )
}
