import { Building2, ChevronRight, MapPin } from 'lucide-react'
import type { JobMatch } from '../types'

interface JobCardProps {
  match: JobMatch
  selected: boolean
  onSelect: () => void
}

export function JobCard({ match, selected, onSelect }: JobCardProps) {
  const matched = new Set([
    ...match.evidence.matched_required,
    ...match.evidence.matched_preferred,
    ...match.evidence.matched_mentioned,
  ])

  return (
    <button
      className={`job-card ${selected ? 'is-selected' : ''}`}
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
    >
      <span className="job-card__rank">{String(match.rank).padStart(2, '0')}</span>
      <span className="job-card__main">
        <span className="job-card__heading">
          <strong>{match.title}</strong>
          <span className="score-pill">{match.predicted_compatibility_score.toFixed(1)}</span>
        </span>
        <span className="job-card__meta">
          <span><Building2 size={14} /> {match.company || 'Company not listed'}</span>
          <span><MapPin size={14} /> {match.location || 'Location not listed'}</span>
        </span>
        <span className="job-card__skills">
          {Array.from(matched).slice(0, 3).map((skill) => <span key={skill}>{skill}</span>)}
          {matched.size > 3 && <span>+{matched.size - 3}</span>}
          {matched.size === 0 && <span className="muted-tag">Limited structured evidence</span>}
        </span>
      </span>
      <ChevronRight className="job-card__chevron" size={18} aria-hidden="true" />
    </button>
  )
}
