import { ArrowUpRight, BriefcaseBusiness, Building2, MapPin, X } from 'lucide-react'
import type { JobMatch } from '../types'
import { ScoreRing } from './ScoreRing'

interface JobDetailProps {
  match: JobMatch
  mobileOpen: boolean
  onClose: () => void
}

function SkillGroup({ title, skills, tone = 'positive', empty }: { title: string; skills: string[]; tone?: 'positive' | 'warning'; empty: string }) {
  return (
    <div className="skill-group">
      <div className="skill-group__header"><h3>{title}</h3><span>{skills.length}</span></div>
      {skills.length > 0 ? (
        <div className="skill-tags">{skills.map((skill) => <span className={`skill-tag skill-tag--${tone}`} key={skill}>{skill}</span>)}</div>
      ) : <p className="empty-copy">{empty}</p>}
    </div>
  )
}

export function JobDetail({ match, mobileOpen, onClose }: JobDetailProps) {
  const evidence = match.evidence
  const matched = Array.from(new Set([...evidence.matched_required, ...evidence.matched_preferred, ...evidence.matched_mentioned]))
  const confidenceLabel = evidence.confidence === 'unavailable' ? 'Evidence unavailable' : `${evidence.confidence} evidence confidence`

  return (
    <aside className={`job-detail ${mobileOpen ? 'is-mobile-open' : ''}`} aria-label="Selected job details">
      <div className="job-detail__mobile-header">
        <strong>Match details</strong>
        <button className="icon-button" type="button" onClick={onClose} aria-label="Close job details"><X size={20} /></button>
      </div>
      <div className="job-detail__top">
        <div className="job-detail__identity">
          <span className="company-mark"><BriefcaseBusiness size={22} /></span>
          <div>
            <div className="rank-label">Rank #{match.rank}</div>
            <h2>{match.title}</h2>
          </div>
        </div>
        <ScoreRing score={match.predicted_compatibility_score} />
      </div>

      <div className="job-facts">
        <span><Building2 size={16} /> {match.company || 'Company not listed'}</span>
        <span><MapPin size={16} /> {match.location || 'Location not listed'}</span>
      </div>

      <div className="compatibility-grid">
        <div><span>Skills compatibility</span><strong>{match.predicted_skills_score.toFixed(1)}<small>/100</small></strong></div>
        <div><span>Evidence coverage</span><strong>{evidence.overlap_ratio == null ? '—' : `${Math.round(evidence.overlap_ratio * 100)}%`}</strong></div>
      </div>

      <div className={`confidence-banner confidence-banner--${evidence.confidence}`}>
        <span className="confidence-dot" />
        <div><strong>{confidenceLabel}</strong><small>{evidence.structured_evidence_available ? 'Based on structured job skills' : 'Job skill fields were not available for a complete comparison'}</small></div>
      </div>

      <div className="job-detail__scroll">
        <SkillGroup title="Matched skills" skills={matched} empty="No structured skill overlap was identified." />
        <SkillGroup title="Missing required skills" skills={evidence.missing_required} tone="warning" empty="No missing required skills were identified from the available evidence." />
        <SkillGroup title="Missing preferred skills" skills={evidence.missing_preferred} tone="warning" empty="No missing preferred skills were identified from the available evidence." />

        <details className="description-panel">
          <summary>Job description</summary>
          <p>{match.description || 'No job description is available.'}</p>
        </details>

        {match.url && (
          <a className="secondary-button" href={match.url} target="_blank" rel="noreferrer">
            View original posting <ArrowUpRight size={16} />
          </a>
        )}

        <details className="model-breakdown">
          <summary>How this match was calculated</summary>
          <p>The production policy combines graph relationships with semantic similarity. These values explain the ranking; they are not hiring probabilities.</p>
          <div><span>Graph score</span><strong>{match.gnn_composite_score.toFixed(3)}</strong></div>
          <div><span>Semantic score</span><strong>{match.calibrated_semantic_score.toFixed(3)}</strong></div>
        </details>
      </div>
    </aside>
  )
}
