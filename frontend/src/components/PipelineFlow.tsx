/**
 * The pipeline's stage flow, read from the backend.
 *
 * Each card is a real stage of the Python pipeline in ../scripts/, with the
 * image that stage actually produced. When no backend is reachable this falls
 * back to naming the stages without previews — it never invents images.
 */
import { useEffect, useState } from 'react'
import { fetchPipelineFlow, isBackendConfigured } from '../services/api'
import type { FlowStage } from '../types'

const STATUS_LABEL: Record<FlowStage['status'], string> = {
  done: 'committed',
  live: 'runs per request',
  unavailable: 'no output',
}

export function PipelineFlow() {
  const [stages, setStages] = useState<FlowStage[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(isBackendConfigured())

  useEffect(() => {
    if (!isBackendConfigured()) {
      setError('No backend connected, so stage previews are unavailable.')
      return
    }

    const controller = new AbortController()

    fetchPipelineFlow(controller.signal)
      .then((response) => setStages(response.stages))
      .catch((cause) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'Could not load the pipeline flow.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [])

  return (
    <aside className="pipeline-info">
      <h3>Pipeline flow</h3>

      {loading && <p className="pipeline-note">Loading stages…</p>}

      {error && <p className="pipeline-note">{error}</p>}

      {stages.length > 0 && (
        <ol className="flow-list">
          {stages.map((stage) => (
            <li key={stage.id} className={`flow-item ${stage.status}`}>
              <div className="flow-thumb">
                {stage.preview_url ? (
                  <img src={stage.preview_url} alt={`${stage.title} output`} loading="lazy" />
                ) : (
                  <span className="placeholder-text">—</span>
                )}
              </div>

              <div className="flow-body">
                <div className="flow-head">
                  <span className="pipeline-index">{stage.step}</span>
                  <strong>{stage.title}</strong>
                  <span className={`flow-badge ${stage.status}`}>
                    {STATUS_LABEL[stage.status]}
                  </span>
                </div>

                {stage.purpose && <p className="flow-purpose">{stage.purpose}</p>}
              </div>
            </li>
          ))}
        </ol>
      )}
    </aside>
  )
}
