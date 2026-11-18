export type EvidenceConfidence = 'high' | 'moderate' | 'limited' | 'unavailable'

export interface SkillEvidence {
  confidence: EvidenceConfidence
  structured_evidence_available: boolean
  required_skill_count: number
  preferred_skill_count: number
  mentioned_skill_count: number
  unique_structured_skill_count: number
  unique_matched_skill_count: number
  overlap_ratio: number | null
  matched_required: string[]
  matched_preferred: string[]
  matched_mentioned: string[]
  missing_required: string[]
  missing_preferred: string[]
}

export interface JobMatch {
  rank: number
  job_id: string
  title: string
  company: string
  location: string
  url: string | null
  description: string
  predicted_compatibility_score: number
  predicted_skills_score: number
  gnn_composite_score: number
  calibrated_semantic_score: number
  semantic_cosine_similarity: number
  evidence: SkillEvidence
}

export interface ResumeAnalysisResponse {
  request_id: string
  api_version: string
  model_version: string
  ranking_policy: string
  processing_time_ms: number
  candidate: {
    detected_skills: string[]
    detected_skill_count: number
    resume_character_count: number
  }
  matches: JobMatch[]
  disclaimer: string
}

export interface ModelInfo {
  model_name: string
  model_version: string
  ranking_policy: string
  gnn_weight: number
  sbert_weight: number
  rankable_job_count: number
  score_meaning: string
  limitations: string[]
}
