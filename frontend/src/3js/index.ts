/**
 * Three.js: the optional 3D tile layer. Everything 3D in the frontend lives
 * in this folder; see README.md for the data it consumes and how it matches
 * the 2D renderer.
 */

export { ThreeTileLayer } from './ThreeTileLayer'
export {
  loadThreeJob,
  projectionFromIntrinsics,
  roomConsistency,
  surfaceGeometry,
  surfaceMaterial,
  TileLayerRenderer,
} from './threeTiles'
export type { RoomGeometry, ThreeJobData, ThreeLayer, ThreeSurfaceRecord } from './threeTiles'
