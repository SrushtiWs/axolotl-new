/**
 * Read-only explainer of what the backend does after Generate is pressed.
 * Mirrors the Python pipeline in ../scripts/. The frontend performs none of
 * these steps — it only collects inputs and displays what the backend returns.
 */
const STEPS = [
  'Object detection',
  'Object extraction',
  'AI creates plain room',
  'Perspective calculated',
  'Real tile projected',
  'Objects restored',
]

export function PipelineInfo() {
  return (
    <aside className="pipeline-info">
      <h3>How your visualization is made</h3>
      <ol className="pipeline-steps">
        {STEPS.map((step, index) => (
          <li key={step}>
            <span className="pipeline-index">{index + 1}</span>
            <span>{step}</span>
          </li>
        ))}
      </ol>
      <p className="pipeline-note">
        Your tile artwork is projected with real perspective geometry, and the
        room&rsquo;s original objects are restored from your photo.
      </p>
    </aside>
  )
}
