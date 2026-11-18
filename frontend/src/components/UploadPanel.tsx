import { useRef, useState, type DragEvent } from 'react'
import { FileText, ShieldCheck, UploadCloud, X } from 'lucide-react'

interface UploadPanelProps {
  file: File | null
  topK: number
  busy: boolean
  onFile: (file: File | null) => void
  onTopK: (value: number) => void
  onAnalyze: () => void
}

function readableBytes(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function UploadPanel({ file, topK, busy, onFile, onTopK, onAnalyze }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const acceptDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files[0]
    if (dropped) onFile(dropped)
  }

  return (
    <section className="upload-panel" aria-labelledby="upload-title">
      <div className="upload-panel__copy">
        <div className="section-kicker">New analysis</div>
        <h1 id="upload-title">Find roles that fit your experience</h1>
        <p>Upload a text-based PDF resume. Your document is analyzed in memory and is not saved by this application.</p>
      </div>

      <div
        className={`drop-zone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={acceptDrop}
      >
        <input
          ref={inputRef}
          className="sr-only"
          type="file"
          accept="application/pdf,.pdf"
          onChange={(event) => onFile(event.target.files?.[0] ?? null)}
          aria-label="Choose PDF resume"
        />
        {file ? (
          <div className="selected-file">
            <span className="selected-file__icon"><FileText size={22} /></span>
            <span className="selected-file__details">
              <strong>{file.name}</strong>
              <small>{readableBytes(file.size)} · PDF document</small>
            </span>
            <button className="icon-button" type="button" onClick={() => onFile(null)} aria-label="Remove selected resume">
              <X size={18} />
            </button>
          </div>
        ) : (
          <button className="drop-zone__button" type="button" onClick={() => inputRef.current?.click()}>
            <span className="drop-zone__icon"><UploadCloud size={26} /></span>
            <span><strong>Drop your resume here</strong><small>or click to browse · PDF up to 10 MB</small></span>
          </button>
        )}
      </div>

      <div className="upload-panel__actions">
        <label className="match-count">
          <span>Results</span>
          <select value={topK} onChange={(event) => onTopK(Number(event.target.value))} disabled={busy}>
            {[5, 10, 15, 20, 25].map((value) => <option key={value} value={value}>Top {value}</option>)}
          </select>
        </label>
        <div className="privacy-note"><ShieldCheck size={16} /> Private, in-memory analysis</div>
        <button className="primary-button" type="button" onClick={onAnalyze} disabled={!file || busy}>
          {busy ? <><span className="spinner" /> Analyzing resume…</> : 'Analyze resume'}
        </button>
      </div>
    </section>
  )
}
