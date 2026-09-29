import { DownloadButton } from './DownloadButton'
import type { GenerateResponse, GenerationStatus, UploadedImage } from '../types'

interface ResultPanelProps {
  status: GenerationStatus
  roomImage: UploadedImage | null
  result: GenerateResponse | null
  message?: string | null
}

export function ResultPanel({ status, roomImage, result, message }: ResultPanelProps) {
  const hasResult = Boolean(result?.result_image_url)

  if (status === 'idle' && !roomImage) {
    return (
      <div className="result-empty">
        <p>Your generated result will appear here.</p>
      </div>
    )
  }

  return (
    <>
      {message && (
        <div className={`notice ${status === 'error' ? 'error' : 'info'}`} role="status">
          {message}
        </div>
      )}

      <div className="result-grid">
        <figure className="result-card">
          <figcaption>Original Room</figcaption>
          <div className="result-image">
            {roomImage ? (
              <img src={roomImage.previewUrl} alt="Original room" />
            ) : (
              <span className="placeholder-text">No room image</span>
            )}
          </div>
        </figure>

        <figure className="result-card">
          <figcaption>Final Result</figcaption>
          <div className="result-image">
            {hasResult ? (
              <img src={result!.result_image_url} alt="Generated tile visualization" />
            ) : (
              <span className="placeholder-text">
                Your generated result will appear here.
              </span>
            )}
          </div>

          {hasResult && (
            <div className="result-actions">
              <DownloadButton
                url={result!.result_image_url}
                filename={`tile-visualization-${result!.job_id ?? 'result'}.png`}
                label="Download result"
              />
            </div>
          )}
        </figure>
      </div>
    </>
  )
}
