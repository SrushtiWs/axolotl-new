// Test-only entry: the app's real Three.js renderer (frontend/src/3js/threeTiles.ts),
// bundled for a bare page so each surface can be drawn at the photo's own
// resolution and read back pixel for pixel. Nothing here re-implements it.
import { loadThreeJob, roomConsistency, TileLayerRenderer } from '../../frontend/src/3js/threeTiles'

declare global {
  interface Window {
    render3d: (jobBase: string, surface: string) => Promise<{ png: string; problems: string[] }>
  }
}

const canvas = document.createElement('canvas')
document.body.appendChild(canvas)
const renderer = new TileLayerRenderer(canvas)

window.render3d = async (jobBase, surface) => {
  const data = await loadThreeJob(`${jobBase}three.json`)
  const problems = roomConsistency(data, surface)
  await renderer.render([{ jobBase, surface, data }])
  return { png: canvas.toDataURL('image/png'), problems }
}
