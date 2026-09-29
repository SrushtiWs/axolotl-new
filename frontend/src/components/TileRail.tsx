/**
 * The product catalogue rail.
 *
 * Search, sort and a list/grid switch over the catalogue, plus the escape
 * hatch that matters most: a customer's own tile artwork. A selected product
 * carries its real millimetre size, which is what the renderer lays out with —
 * so picking a product changes the tile scale, not just the picture.
 */

import { useMemo, useState } from 'react'
import type { CatalogueTile } from '../types'

interface Props {
  tiles: CatalogueTile[]
  selectedId?: string | null
  onSelect: (tile: CatalogueTile) => void
  onUploadClick: () => void
}

type Sort = 'new' | 'name' | 'size'

export function TileRail({ tiles, selectedId, onSelect, onUploadClick }: Props) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<Sort>('new')
  const [grid, setGrid] = useState(false)

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase()

    const filtered = needle
      ? tiles.filter(
          (tile) =>
            tile.name.toLowerCase().includes(needle) ||
            tile.sku.includes(needle) ||
            tile.size.toLowerCase().includes(needle),
        )
      : tiles

    const sorted = [...filtered]

    if (sort === 'name') sorted.sort((a, b) => a.name.localeCompare(b.name))
    if (sort === 'size') sorted.sort((a, b) => a.width * a.height - b.width * b.height)

    return sorted
  }, [tiles, query, sort])

  return (
    <div className="rail-body">
      <div className="rail-search">
        <span aria-hidden="true">🔍</span>
        <input
          value={query}
          placeholder="Search by name or code"
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>

      <div className="rail-tools">
        <button type="button" className="btn tiny">⚙ Filters</button>
        <div className="view-toggle">
          <button
            type="button"
            className={grid ? 'view' : 'view active'}
            onClick={() => setGrid(false)}
            title="List"
          >
            ☰
          </button>
          <button
            type="button"
            className={grid ? 'view active' : 'view'}
            onClick={() => setGrid(true)}
            title="Grid"
          >
            ▦
          </button>
        </div>
      </div>

      <label className="rail-sort">
        <span aria-hidden="true">⇅</span>
        <select value={sort} onChange={(event) => setSort(event.target.value as Sort)}>
          <option value="new">New Products First</option>
          <option value="name">Name A–Z</option>
          <option value="size">Smallest Size</option>
        </select>
      </label>

      <button type="button" className="rail-upload" onClick={onUploadClick}>
        ⤒ Use my own tile image
      </button>

      <div className={grid ? 'rail-list grid' : 'rail-list'}>
        {shown.map((tile) => (
          <button
            key={tile.id}
            type="button"
            className={tile.id === selectedId ? 'product active' : 'product'}
            onClick={() => onSelect(tile)}
          >
            <img src={tile.thumb} alt={tile.name} loading="lazy" />

            <div className="product-meta">
              <strong>{tile.name.toUpperCase()}</strong>
              <small>SKU: {tile.sku}</small>
              <small>Size: {tile.size}</small>
              <small>{tile.finish}</small>
              {tile.badge && (
                <span className={`tag ${tile.badge.toLowerCase().replace(/\s+/g, '-')}`}>
                  {tile.badge}
                </span>
              )}
            </div>
          </button>
        ))}

        {shown.length === 0 && <p className="rail-empty">No products match “{query}”.</p>}
      </div>
    </div>
  )
}
