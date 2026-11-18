interface ScoreRingProps {
  score: number
  size?: 'small' | 'large'
  label?: string
}

export function ScoreRing({ score, size = 'large', label = 'Compatibility' }: ScoreRingProps) {
  const safe = Math.max(0, Math.min(100, score))
  const radius = 44
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (safe / 100) * circumference

  return (
    <div className={`score-ring score-ring--${size}`} aria-label={`${label}: ${safe.toFixed(1)} out of 100`}>
      <svg viewBox="0 0 104 104" aria-hidden="true">
        <circle className="score-ring__track" cx="52" cy="52" r={radius} />
        <circle
          className="score-ring__value"
          cx="52"
          cy="52"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <span className="score-ring__text">
        <strong>{safe.toFixed(1)}</strong>
        <small>/100</small>
      </span>
    </div>
  )
}
