import { useMemo, useState } from 'react'
import { useStore } from '../store/store.js'
import { buildLayerFlow } from '../lib/layerFlow.js'
import { isMoeLayer } from '../lib/model.js'
import { DTYPE_BYTES, formatBytes, formatCount, formatShape, prod } from '../lib/format.js'
import { GROUPS, rankCoords, worldSize } from '../lib/mesh.js'
import { placeParam, placementLabel, shardLevels } from '../lib/shard.js'
import { AXIS_COLOR } from '../lib/colors.js'
import { Segmented, Toggle } from './Fields.jsx'
import { Stat } from './ParamTable.jsx'

const LANE = 252
const NODE_W = 226
const RAIL = 64
const GAP = 46
const PAD = 16
const LH = 13
const RAIL_R = 28 // right-hand rail for long skip edges

function nodeHeight(n) {
  switch (n.kind) {
    case 'fork':
      return 12
    case 'add':
      return 30
    case 'io':
      return 44
    case 'comm':
      return 46 + 34 * Math.max(n.fwd.length, n.bwd.length, 1)
    default:
      return 48 + 38 * (n.params?.length ?? 0)
  }
}

function layout(graph) {
  const rows = new Map()
  for (const n of graph.nodes) {
    if (!rows.has(n.row)) rows.set(n.row, [])
    rows.get(n.row).push(n)
  }
  const pos = new Map()
  let y = PAD
  for (const r of [...rows.keys()].sort((a, b) => a - b)) {
    const ns = rows.get(r)
    const hMax = Math.max(...ns.map(nodeHeight))
    for (const n of ns) {
      const h = nodeHeight(n)
      const w = n.kind === 'fork' ? 12 : n.kind === 'add' ? 30 : NODE_W
      const cx = RAIL + (n.col + 0.5) * LANE
      pos.set(n.id, { x: cx - w / 2, y: y + (hMax - h) / 2, w, h, cx, rowTop: y, rowBottom: y + hMax })
    }
    y += hMax + GAP
  }
  const lanes = Math.max(3, Math.ceil(Math.max(...graph.nodes.map((n) => n.col)) + 1))
  const rail = graph.edges.some((e) => e.route === 'rail')
  const railX = RAIL + lanes * LANE + RAIL_R / 2
  return { pos, width: RAIL + lanes * LANE + (rail ? RAIL_R : PAD), height: y - GAP + PAD, railX }
}

const labelText = (l, mode) => `${l.name ? `${l.name} ` : ''}${fmtShape(l.shape, mode)}`
const labelWidth = (labels, mode) => Math.max(0, ...labels.map((l) => labelText(l, mode).length + 1)) * 6.4

function edgeGeometry(graph, pos, width, mode, railX) {
  const inCount = new Map()
  const outCount = new Map()
  for (const e of graph.edges) {
    if (e.kind === 'residual' || e.route === 'rail') continue
    inCount.set(e.to, (inCount.get(e.to) ?? 0) + 1)
    outCount.set(e.from, (outCount.get(e.from) ?? 0) + 1)
  }
  return graph.edges.map((e) => {
    const s = pos.get(e.from)
    const t = pos.get(e.to)
    if (e.kind === 'residual') {
      const x0 = s.x
      const y0 = s.y + s.h / 2
      const x1 = t.x
      const y1 = t.y + t.h / 2
      const rx = RAIL / 2
      const r = 10
      return {
        ...e,
        d: `M ${x0} ${y0} L ${rx + r} ${y0} Q ${rx} ${y0} ${rx} ${y0 + r} L ${rx} ${y1 - r} Q ${rx} ${y1} ${rx + r} ${y1} L ${x1} ${y1}`,
        label: { x: rx - 6, y: (y0 + y1) / 2 },
      }
    }
    if (e.route === 'rail') {
      const x0 = s.x + s.w
      const y0 = s.y + s.h / 2
      const x1 = t.x + t.w
      const y1 = t.y + t.h / 2
      const r = 10
      return {
        ...e,
        d: `M ${x0} ${y0} L ${railX - r} ${y0} Q ${railX} ${y0} ${railX} ${y0 + r} L ${railX} ${y1 - r} Q ${railX} ${y1} ${railX - r} ${y1} L ${x1} ${y1}`,
        label: { x: railX, y: y0, anchor: 'middle' },
      }
    }
    const sx = s.cx
    const sy = s.y + s.h
    const tx = t.cx
    const ty = t.y
    let d
    let mid
    if (Math.abs(sx - tx) < 1) {
      d = `M ${sx} ${sy} L ${tx} ${ty}`
      mid = { x: sx, y: (sy + ty) / 2 }
    } else if (e.route === 'early') {
      // Bend in the gap right below the source row, then run down the target lane.
      const y0 = Math.max(sy, s.rowBottom)
      const y1 = Math.min(y0 + GAP - 6, ty)
      const ym = (y0 + y1) / 2
      d = `M ${sx} ${sy} L ${sx} ${y0} C ${sx} ${ym} ${tx} ${ym} ${tx} ${y1} L ${tx} ${ty}`
      mid = { x: (sx + tx) / 2, y: ym }
    } else {
      // Bend only inside the gap above the target's row, then drop straight in.
      const yr = Math.min(t.rowTop, ty)
      const y0 = Math.max(sy, yr - GAP + 6)
      const ym = (y0 + yr) / 2
      d = `M ${sx} ${sy} L ${sx} ${y0} C ${sx} ${ym} ${tx} ${ym} ${tx} ${yr} L ${tx} ${ty}`
      mid = y0 - sy > 60 ? { x: sx, y: (sy + y0) / 2 } : { x: (sx + tx) / 2, y: ym }
    }
    const where = e.labelAt ?? (inCount.get(e.to) === 1 ? 'dst' : outCount.get(e.from) === 1 ? 'src' : 'mid')
    const n = e.labels.length
    const w = labelWidth(e.labels, mode)
    // Put the label on the side of the line that no curve bends toward.
    const pickSide = (x, bendsRight, bendsLeft) => {
      let left = bendsRight && !bendsLeft
      if (!left && x + 8 + w > width) left = true
      if (left && x - 8 - w < RAIL / 2 + 10) left = false
      return left ? { x: x - 8, anchor: 'end' } : { x: x + 8, anchor: 'start' }
    }
    const siblings = graph.edges.filter((o) => o !== e && o.kind !== 'residual' && o.route !== 'rail')
    let label
    if (where === 'dst') {
      const fromRight = sx > tx + 1 || siblings.some((o) => o.to === e.to && pos.get(o.from).cx > tx + 1)
      const fromLeft = sx < tx - 1
      label = { ...pickSide(tx, fromRight, fromLeft), y: Math.min(t.rowTop, ty) - 8 - (n - 1) * LH }
    } else if (where === 'src') {
      const right = tx > sx + 1 || siblings.some((o) => o.from === e.from && pos.get(o.to).cx > sx + 1)
      label = { ...pickSide(sx, right, tx < sx - 1), y: Math.max(sy, s.rowBottom) + 14 }
    } else if (Math.abs(sx - tx) < 1 || mid.x === sx) {
      label = { ...pickSide(mid.x, false, false), y: mid.y - ((n - 1) * LH) / 2 + 4 }
    } else {
      label = { x: mid.x, anchor: 'middle', y: mid.y - 7 - (n - 1) * LH }
    }
    return { ...e, d, label }
  })
}

const fmtShape = (shape, mode) =>
  `[${shape.map((x) => (mode === 'symbols' ? x.sym : x.n.toLocaleString('en-US'))).join(', ')}]`
const compactShape = (shape) => `[${shape.join(', ')}]`
const shortName = (fqn) => {
  const parts = fqn.split('.')
  const last = parts[parts.length - 1]
  return last === 'weight' || last === 'bias' ? parts.slice(-2).join('.') : last
}

export default function LayerView({ params, coords }) {
  const model = useStore((s) => s.model)
  const par = useStore((s) => s.parallel)
  const train = useStore((s) => s.train)
  const prec = useStore((s) => s.precision)
  const [layerSel, setLayerSel] = useState(null)
  const [dir, setDir] = useState('fwd')
  const [labelMode, setLabelMode] = useState('numbers')
  const [animate, setAnimate] = useState(true)
  const [selId, setSelId] = useState(null)

  const defaultLayer = model.arch === 'moe' ? Math.min(model.n_dense_layers, model.n_layers - 1) : 0
  const li = Math.min(layerSel ?? defaultLayer, model.n_layers - 1)

  const graph = useMemo(() => buildLayerFlow(model, par, train, prec, params, li, coords), [model, par, train, prec, params, li, coords])
  const geo = useMemo(() => {
    const lay = layout(graph)
    return { ...lay, edges: edgeGeometry(graph, lay.pos, lay.width, labelMode, lay.railX) }
  }, [graph, labelMode])
  const byFqn = useMemo(() => new Map(params.map((p) => [p.fqn, p])), [params])
  const selected = graph.nodes.find((n) => n.id === selId) ?? null

  const stats = useMemo(() => {
    let g = 0
    let l = 0
    for (const p of graph.layerParams) {
      if (p.kind !== 'param') continue
      g += prod(p.shape)
      l += placeParam(p, par, coords).localNumel
    }
    return { g, l }
  }, [graph, par, coords])

  const axes = [...new Set(graph.nodes.flatMap((n) => [...n.fwd, ...n.bwd].map((e) => e.axis)))]
  const bwd = dir === 'bwd'

  return (
    <div className="layer-view">
      <div className="lf-controls">
        <label className="field lf-layer">
          <span className="field-label">Which block (i)</span>
          <select value={li} onChange={(e) => { setLayerSel(Number(e.target.value)); setSelId(null) }}>
            {Array.from({ length: model.n_layers }, (_, i) => (
              <option key={i} value={i}>
                i = {i} · {isMoeLayer(model, i) ? 'MoE' : 'dense'}
              </option>
            ))}
          </select>
        </label>
        <div className="lf-ctl">
          <span className="field-label">Pass</span>
          <Segmented value={dir} onChange={setDir} options={{ fwd: 'Forward ↓', bwd: 'Backward ↑' }} />
        </div>
        <div className="lf-ctl">
          <span className="field-label">Shapes</span>
          <Segmented value={labelMode} onChange={setLabelMode} options={{ numbers: 'Numbers', symbols: 'Symbols' }} />
        </div>
        <Toggle label="animate flow" checked={animate} onChange={setAnimate} />
      </div>

      <div className="stat-row">
        <Stat label={`layers.${li} params`} value={formatCount(stats.g)} sub={graph.moe ? 'MoE block' : 'dense block'} />
        <Stat label={`On rank ${coords.rank}`} value={formatCount(stats.l)} sub={`${formatBytes(stats.l * DTYPE_BYTES[prec.weight])} weights (${prec.weight})`} />
        <Stat label={`Forward comm · ${graph.totals.fwdCalls} calls`} value={formatBytes(graph.totals.fwd)} sub="this block only" />
        <Stat label={`Backward comm · ${graph.totals.bwdCalls} calls`} value={formatBytes(graph.totals.bwd)} sub="this block only" />
      </div>

      <div className="lf-wrap">
        <div className="card lf-card">
          <div className="card-head">
            <div>
              <h2>
                {bwd ? 'Backward' : 'Forward'} pass · layers.{li} on rank {coords.rank}
              </h2>
              <p className="card-sub">
                {bwd
                  ? 'Gradients flow bottom → top. Edges carry ∂ of the activation; param nodes show ∂W on this rank; collectives are the backward conjugates.'
                  : 'Activations flow top → bottom. Edges show local tensor shapes on this rank; collectives appear where they fire.'}{' '}
                The block is drawn as a generic <code>layers[i]</code>, with the embedding above it and the model head
                below it. Click a node for details.
              </p>
            </div>
            <div className="legend">
              {axes.map((a) => (
                <span key={a} className="legend-item">
                  <span className="swatch" style={{ background: AXIS_COLOR[a] }} />
                  {a}
                </span>
              ))}
              <span className="legend-item">
                <span className={`line-key ${bwd ? 'grad' : 'act'}`} />
                {bwd ? 'gradient' : 'activation'}
              </span>
            </div>
          </div>
          {labelMode === 'symbols' && (
            <p className="lf-symbols mono faint">
              b batch · s seq · d dim · h heads · kv kv-heads · hd head dim · ffn ffn dim · E experts · k top-k · cp / tp degrees
            </p>
          )}
          <div className="lf-scroll">
            <div className="lf-canvas" style={{ width: geo.width, height: geo.height }}>
              <svg width={geo.width} height={geo.height} aria-hidden="true">
                <defs>
                  <marker id="lf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" className={bwd ? 'lf-arrow grad' : 'lf-arrow act'} />
                  </marker>
                </defs>
                {geo.edges.map((e, i) => {
                  // nothing flows back along an integer edge: draw it dead, no arrow
                  const dead = bwd && e.noGrad
                  const markers = dead ? {} : bwd ? { markerStart: 'url(#lf-arrow)' } : { markerEnd: 'url(#lf-arrow)' }
                  return (
                    <g key={i}>
                      <path d={e.d} className={`lf-edge ${e.kind === 'residual' ? 'residual' : ''} ${bwd ? 'grad' : 'act'} ${dead ? 'nograd' : ''}`} {...markers} />
                      {animate && !dead && <path d={e.d} className={`lf-flow ${bwd ? 'grad rev' : 'act'}`} />}
                    </g>
                  )
                })}
                {geo.edges.map((e, i) =>
                  e.kind === 'residual' ? (
                    <text key={`l${i}`} className="lf-label" x={e.label.x} y={e.label.y} textAnchor="middle" transform={`rotate(-90 ${e.label.x} ${e.label.y})`}>
                      residual{bwd ? ' (∂ copied)' : ''}
                    </text>
                  ) : e.labels.length ? (
                    <text key={`l${i}`} className="lf-label" x={e.label.x} y={e.label.y} textAnchor={e.label.anchor}>
                      {e.labels.map((l, j) => (
                        <tspan key={j} x={e.label.x} dy={j ? LH : 0}>
                          {bwd && e.noGrad
                            ? `${l.name ? `${l.name} · ` : ''}no grad`
                            : `${bwd ? '∂' : ''}${l.name ? `${l.name} ` : bwd ? ' ' : ''}${fmtShape(l.shape, labelMode)}`}
                        </tspan>
                      ))}
                    </text>
                  ) : null,
                )}
              </svg>
              {graph.nodes.map((n) => (
                <NodeBox
                  key={n.id}
                  n={n}
                  box={geo.pos.get(n.id)}
                  dir={dir}
                  sel={n.id === selId}
                  onSelect={() => setSelId(n.id === selId ? null : n.id)}
                  byFqn={byFqn}
                  par={par}
                  coords={coords}
                  prec={prec}
                />
              ))}
            </div>
          </div>
        </div>

        <Detail node={selected} graph={graph} li={li} dir={dir} par={par} coords={coords} prec={prec} byFqn={byFqn} onClear={() => setSelId(null)} />
      </div>
    </div>
  )
}

function NodeBox({ n, box, dir, sel, onSelect, byFqn, par, coords, prec }) {
  const style = { left: box.x, top: box.y, width: box.w, height: box.h }
  if (n.kind === 'fork') return <div className="lf-node lf-fork" style={style} title="residual branch" />
  if (n.kind === 'add') return <div className="lf-node lf-add" style={style}>+</div>

  const interactive = {
    role: 'button',
    tabIndex: 0,
    onClick: onSelect,
    onKeyDown: (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        onSelect()
      }
    },
  }
  const selCls = sel ? 'sel' : ''

  if (n.kind === 'io') {
    const dead = dir === 'bwd' && n.noGrad
    return (
      <div className={`lf-node lf-io ${selCls} ${dead ? 'dimmed' : ''}`} style={style} {...interactive}>
        <div className="lf-title mono">{n.title}</div>
        <div className="lf-sub">{dead ? 'no gradient · backward stops here' : n.sub}</div>
      </div>
    )
  }

  if (n.kind === 'comm') {
    const entries = n[dir]
    const other = n[dir === 'fwd' ? 'bwd' : 'fwd']
    const accent = AXIS_COLOR[(entries[0] ?? other[0])?.axis] ?? 'var(--border-strong)'
    const dimmed = entries.length === 0 || entries.every((e) => e.idle)
    return (
      <div className={`lf-node lf-comm ${selCls} ${dimmed ? 'dimmed' : ''}`} style={{ ...style, '--lf-accent': accent }} {...interactive}>
        <div className="lf-title">{n.title}</div>
        <div className="lf-sub">{n.sub}</div>
        {entries.length === 0 && <div className="lf-comm-line faint">nothing in {dir === 'fwd' ? 'forward' : 'backward'}</div>}
        {entries.map((e, i) => (
          <div key={i} className={`lf-comm-line ${e.idle ? 'idle' : ''}`}>
            <div className="lf-comm-top">
              <span className="lf-chip" style={{ background: AXIS_COLOR[e.axis] }}>
                {e.axis}
              </span>
              <b>{e.op}</b>
              {e.count > 1 && <span className="faint">×{e.count}</span>}
            </div>
            <div className="lf-comm-sub mono">
              {e.group}
              {e.idle ? '' : ` · ${formatBytes(e.bytes)}`}
            </div>
          </div>
        ))}
      </div>
    )
  }

  // module / op
  const ps = (n.params ?? []).map((f) => byFqn.get(f)).filter(Boolean)
  const levelAxes = ps.length ? [...new Set(shardLevels(ps[0], par).filter((l) => l.kind !== 'replicate').map((l) => l.axis))] : []
  const accent = levelAxes.includes('EP') ? AXIS_COLOR.EP : levelAxes.includes('TP') ? AXIS_COLOR.TP : 'var(--border-strong)'
  const bytes = DTYPE_BYTES[dir === 'bwd' ? prec.grad : prec.weight]
  return (
    <div className={`lf-node lf-${n.kind} ${selCls}`} style={{ ...style, '--lf-accent': accent }} {...interactive}>
      <div className="lf-title">
        <span className="mono lf-name">{n.title}</span>
        <span className="lf-chips">
          {levelAxes.map((a) => (
            <span key={a} className="lf-chip" style={{ background: AXIS_COLOR[a] }}>
              {a}
            </span>
          ))}
        </span>
      </div>
      <div className="lf-sub">{n.sub}</div>
      {ps.map((p) => {
        const pl = placeParam(p, par, coords)
        const noGrad = dir === 'bwd' && p.kind === 'buffer'
        return (
          <div key={p.fqn} className="lf-param">
            <div className="lf-param-top mono">
              <span>
                {dir === 'bwd' && !noGrad ? '∂' : ''}
                {shortName(p.fqn)}
              </span>
              <span className="num">{noGrad ? 'no grad' : formatBytes(pl.localNumel * bytes)}</span>
            </div>
            <div className="lf-param-shape mono">
              {compactShape(p.shape)} → {pl.present ? compactShape(pl.localShape) : 'absent'}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Detail({ node, graph, li, dir, par, coords, prec, byFqn, onClear }) {
  const wB = DTYPE_BYTES[prec.weight]
  const gB = DTYPE_BYTES[prec.grad]

  if (!node) {
    const rows = graph.layerParams
      .map((p) => ({ p, pl: placeParam(p, par, coords) }))
      .filter(({ p, pl }) => !(p.expert && p.expert.index != null && !pl.present))
    return (
      <div className="card lf-detail">
        <h2>layers.{li} parameters</h2>
        <p className="card-sub">Local sizes on rank {coords.rank}. Click a node in the diagram for placement and collectives.</p>
        <div className="table-scroll">
          <table className="ptable">
            <thead>
              <tr>
                <th>FQN</th>
                <th className="right">Global</th>
                <th className="right">Local</th>
                <th className="right">Weights</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ p, pl }) => (
                <tr key={p.fqn}>
                  <td className="mono">{p.fqn.replace(`layers.${li}.`, '')}</td>
                  <td className="mono num right">{formatCount(prod(p.shape))}</td>
                  <td className="mono num right">{formatCount(pl.localNumel)}</td>
                  <td className="mono num right">{formatBytes(pl.localNumel * wB)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const ps = (node.params ?? []).map((f) => byFqn.get(f)).filter(Boolean)
  return (
    <div className="card lf-detail">
      <div className="lf-detail-head">
        <h2 className="mono">{node.title}</h2>
        <button type="button" className="btn-ghost" onClick={onClear} aria-label="Close details">
          ✕
        </button>
      </div>
      {node.sub && <p className="card-sub">{node.sub}</p>}

      {ps.map((p) => {
        const levels = shardLevels(p, par)
        const pl = placeParam(p, par, coords, levels)
        return (
          <div key={p.fqn} className="lf-dblock">
            <code className="accent">{p.fqn}</code>
            <dl className="lf-kv mono">
              <dt>global</dt>
              <dd>{formatShape(p.shape)}</dd>
              <dt>local</dt>
              <dd>{pl.present ? formatShape(pl.localShape) : 'absent on this rank'}</dd>
              <dt>placement</dt>
              <dd>{placementLabel(levels)}</dd>
              <dt>weight</dt>
              <dd>
                {formatBytes(pl.localNumel * wB)} · {prec.weight}
              </dd>
              {p.kind === 'param' && (
                <>
                  <dt>grad</dt>
                  <dd>
                    {formatBytes(pl.localNumel * gB)} · {prec.grad}
                  </dd>
                </>
              )}
            </dl>
          </div>
        )
      })}

      {node.kind === 'comm' &&
        ['fwd', 'bwd'].map((d) => (
          <div key={d} className={`lf-dblock ${d === dir ? '' : 'lf-dim'}`}>
            <h3 className="lf-dh">{d === 'fwd' ? 'Forward' : 'Backward'}</h3>
            {node[d].length === 0 && <p className="faint lf-note">nothing</p>}
            {node[d].map((e, i) => (
              <CommDetail key={i} e={e} par={par} coords={coords} />
            ))}
          </div>
        ))}

      {ps.length === 0 && node.kind !== 'comm' && <p className="faint lf-note">No parameters — pure compute on the local shard.</p>}
    </div>
  )
}

function CommDetail({ e, par, coords }) {
  const W = worldSize(par)
  let ranks = null
  if (e.groupKey && GROUPS[e.groupKey]) {
    ranks = []
    for (let r = 0; r < W && ranks.length < 65; r++) if (GROUPS[e.groupKey].test(rankCoords(par, r), coords, par)) ranks.push(r)
  }
  return (
    <div className="lf-dcomm">
      <div className="lf-comm-top">
        <span className="lf-chip" style={{ background: AXIS_COLOR[e.axis] }}>
          {e.axis}
        </span>
        <b>{e.op}</b>
        {e.count > 1 && <span className="faint">×{e.count}</span>}
      </div>
      <dl className="lf-kv mono">
        <dt>group</dt>
        <dd>{e.group}</dd>
        {!e.idle && (
          <>
            <dt>buffer</dt>
            <dd>
              {formatBytes(e.bytes)}
              {e.count > 1 ? ` × ${e.count} = ${formatBytes(e.bytes * e.count)}` : ''}
            </dd>
          </>
        )}
        {ranks && (
          <>
            <dt>ranks</dt>
            <dd>
              {ranks.slice(0, 64).join(', ')}
              {ranks.length > 64 ? ', …' : ''}
            </dd>
          </>
        )}
      </dl>
      <p className="lf-note">{e.note}</p>
    </div>
  )
}
