/**
 * The company lockup: mark, name, and what this tool is.
 *
 * The mark is loaded from `/brand/logo.svg` rather than being drawn here, so
 * replacing the artwork is a file swap and needs no code change. If that file
 * is missing the image is hidden rather than showing a broken-image icon, and
 * the wordmark still reads correctly on its own.
 */

interface Props {
  /** `compact` is the studio rail; `full` is the landing screen. */
  size?: 'compact' | 'full'
}

export function Brand({ size = 'compact' }: Props) {
  return (
    <div className={size === 'full' ? 'brand brand-full' : 'brand'}>
      <img
        className="brand-mark-img"
        src="/brand/logo.svg"
        alt=""
        onError={(event) => {
          event.currentTarget.style.display = 'none'
        }}
      />
      <span className="brand-text">
        <strong>Rich International</strong>
        <small>Visualizer</small>
      </span>
    </div>
  )
}
