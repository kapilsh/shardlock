import { useMemo, useState } from 'react'
import { DTYPE_BYTES, formatBytes, formatCount, formatShape } from '../lib/format.js'
import { AXIS_COLOR } from '../lib/colors.js'
import { countRanksForPath, placeParam, ranksForPath, shardCells, shardLevels } from '../lib/shard.js'

const MAX_W = 620
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x))

// Which tensor dims map to the x / y axes of the drawing.
function axesFor(p, levels) {
  const n = p.shape.length
  if (n === 1) return { x: 0, y: null, depth: null }
  if (n === 2) return { x: 1, y: 0, depth: null }
  const tpLv = levels.find((l) => l.axis === 'TP')
  const y = tpLv && tpLv.dim > 0 ? tpLv.dim : 1
  return { x: 0, y, depth: [1, 2].find((d) => d !== y) }
}

export default function TensorDiagram({ p, par, coords, weightDtype }) {
  const [hover, setHover] = useState(null)
  const levels = useMemo(() => shardLevels(p, par), [p, par])
  const { levels: shardLvls, depth, cells } = useMemo(() => shardCells(p, par, 512), [p, par])
  const place = placeParam(p, par, coords, levels)
  const ax = axesFor(p, levels)

  const X = p.shape[ax.x]
  const Y = ax.y == null ? 1 : p.shape[ax.y]
  let w = MAX_W
  let h = ax.y == null ? 34 : clamp((w * Y) / X, 60, 300)
  if (ax.y != null && (w * Y) / X > 300) w = clamp((300 * X) / Y, 140, MAX_W)
  const sx = w / X
  const sy = h / Y
  const rectOf = (ranges) => ({
    x: ranges[ax.x].start * sx,
    y: ax.y == null ? 0 : ranges[ax.y].start * sy,
    w: ranges[ax.x].len * sx,
    h: ax.y == null ? h : ranges[ax.y].len * sy,
  })

  // Outline boxes per level: union of leaf cells sharing a path prefix.
  const outlines = useMemo(() => {
    const out = []
    for (let L = 0; L < depth; L++) {
      const groups = new Map()
      for (const c of cells) {
        const key = c.path.slice(0, L + 1).join('.')
        const g = groups.get(key)
        if (!g) groups.set(key, c.ranges.map((r) => ({ lo: r.start, hi: r.start + r.len })))
        else c.ranges.forEach((r, d) => { g[d].lo = Math.min(g[d].lo, r.start); g[d].hi = Math.max(g[d].hi, r.start + r.len) })
      }
      for (const [key, g] of groups) out.push({ L, key, ranges: g.map((r) => ({ start: r.lo, len: r.hi - r.lo })) })
    }
    return out
  }, [cells, depth])

  const selPath = shardLvls.slice(0, depth).map((l) => l.indexOf(coords))
  const hoverInfo = hover
    ? {
        n: countRanksForPath(p, par, shardLvls, hover.path),
        sample: ranksForPath(p, par, shardLvls, hover.path, 6),
      }
    : null

  // Step-by-step shapes for this rank.
  const steps = levels.map((lv, i) => ({ lv, shape: placeParam(p, par, coords, levels.slice(0, i + 1)).localShape }))
  const bytes = DTYPE_BYTES[weightDtype]

  return (
    <div className="tensor">
      <div className="tensor-head">
        <code className="tensor-fqn">{p.fqn}</code>
        <span className="faint mono">
          {p.kind === 'buffer' ? 'buffer' : 'parameter'} · {formatCount(p.shape.reduce((a, b) => a * b, 1))} elements
        </span>
      </div>

      <div className="tensor-chain mono">
        <span className="shape-pill">global {formatShape(p.shape)}</span>
        {steps.map(({ lv, shape }, i) => (
          <span key={i} className="chain-step">
            <span className="arrow" style={{ color: AXIS_COLOR[lv.axis] }}>
              →{' '}
              {lv.kind === 'replicate'
                ? `${lv.axis} replicate ×${lv.degree}`
                : lv.kind === 'place'
                  ? `EP place on ep${lv.owner}`
                  : `${lv.axis} Shard(${lv.dim}) /${lv.degree}${lv.replicas ? ` (×${lv.replicas} replicas)` : ''}`}
            </span>
            <span className="shape-pill">{shape ? formatShape(shape) : 'absent'}</span>
          </span>
        ))}
      </div>

      <div className="tensor-canvas">
        <div className="tensor-axis-y mono faint">{ax.y != null ? `dim ${ax.y} (${Y.toLocaleString('en-US')})` : ''}</div>
        <div>
          <svg width={w} height={h} className="tensor-svg" onMouseLeave={() => setHover(null)} role="img" aria-label={`Shard layout of ${p.fqn}`}>
            <rect x={0} y={0} width={w} height={h} className="tensor-bg" />
            {place.present && (
              <rect {...rectAttrs(rectOf(place.ranges))} className="tensor-local" />
            )}
            {cells.map((c) => {
              const r = rectOf(c.ranges)
              const isSel = c.path.every((v, i) => v === selPath[i])
              return (
                <rect
                  key={c.path.join('.')}
                  {...rectAttrs(r)}
                  className={`tensor-cell ${isSel ? 'sel' : ''}`}
                  onMouseEnter={() => setHover({ path: c.path, rect: r })}
                />
              )
            })}
            {outlines.map((o) => (
              <rect
                key={`${o.L}:${o.key}`}
                {...rectAttrs(rectOf(o.ranges))}
                fill="none"
                stroke={AXIS_COLOR[shardLvls[o.L].axis]}
                strokeWidth={Math.max(1, 3 - o.L)}
                pointerEvents="none"
              />
            ))}
            {!place.present && (
              <text x={w / 2} y={h / 2} textAnchor="middle" dominantBaseline="middle" className="tensor-absent">
                not on this rank
              </text>
            )}
          </svg>
          <div className="tensor-axis-x mono faint">
            dim {ax.x} ({X.toLocaleString('en-US')}){ax.depth != null ? ` · depth: dim ${ax.depth} (${p.shape[ax.depth].toLocaleString('en-US')})` : ''}
          </div>
        </div>
        <div className="tensor-side">
          <div className="legend">
            {shardLvls.map((l, i) => (
              <span key={i} className="legend-item">
                <span className="line-key" style={{ background: AXIS_COLOR[l.axis] }} />
                {l.axis} ×{l.degree} (dim {l.dim}){i >= depth ? ' — not drawn' : ''}
              </span>
            ))}
            <span className="legend-item">
              <span className="swatch local" />
              rank {coords.rank} local shard
            </span>
          </div>
          <div className="tensor-local-stat">
            <div className="stat-label">Local on rank {coords.rank}</div>
            <div className="mono">{place.present ? formatShape(place.localShape) : 'absent'}</div>
            <div className="mono faint">
              {formatCount(place.localNumel)} el · {formatBytes(place.localNumel * bytes)} ({weightDtype})
            </div>
          </div>
          {hover && hoverInfo && (
            <div className="tensor-hover mono">
              <div>
                shard {hover.path.map((v, i) => `${shardLvls[i].axis}${v}`).join(' · ') || 'whole tensor'}
              </div>
              <div className="faint">
                held by {hoverInfo.n.toLocaleString('en-US')} rank{hoverInfo.n === 1 ? '' : 's'}: {hoverInfo.sample.join(', ')}
                {hoverInfo.n > hoverInfo.sample.length ? ', …' : ''}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function rectAttrs(r) {
  return { x: r.x, y: r.y, width: Math.max(0, r.w), height: Math.max(0, r.h) }
}
