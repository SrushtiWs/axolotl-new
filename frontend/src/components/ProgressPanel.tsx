import type { PipelineStep } from '../types'

interface ProgressPanelProps {
  steps: PipelineStep[]
}

export function ProgressPanel({ steps }: ProgressPanelProps) {
  return (
    <div className="progress-panel" role="status" aria-live="polite">
      <h3 className="progress-title">Generating Tile Visualization...</h3>
      <ul className="progress-steps">
        {steps.map((step) => (
          <li key={step.id} className={`progress-step ${step.status}`}>
            <span className="step-marker" aria-hidden="true">
              {step.status === 'done' ? '✓' : step.status === 'active' ? '●' : '○'}
            </span>
            <span className="step-label">{step.label}</span>
          </li>
        ))}
      </ul>
      <p className="progress-wait">Please wait...</p>
    </div>
  )
}
