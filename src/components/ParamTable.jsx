import { Fragment, useMemo, useState } from 'react'
import { useStore } from '../store/store.js'
import { DTYPE_BYTES, formatBytes, formatCount, formatShape, prod } from '../lib/format.js'
import { GROUP_LABELS } from '../lib/model.js'
import { OPTIMIZERS } from '../lib/memory.js'
import { placeParam, placementLabel, shardLevels } from '../lib/shard.js'
import TensorDiagram from './TensorDiagram.jsx'

function buildTree(rows) {
  const root = { name: '', path: '', children: new Map(), leaves: [] }
  for (const r of rows) {
    const parts = r.p.fqn.split('.')
    let node = root
    for (let i = 0; i < parts.length - 1; i++) {
      const path = parts.slice(0, i + 1).join('.')
      if (!node.children.has(parts[i])) node.children.set(parts[i], { name: parts[i], path, children: new Map(), leaves: [] })
      node = node.children.get(parts[i])
    }
    node.leaves.push({ ...r, name: parts[parts.length - 1] })
  }
  const agg = (n) => {
    let g = 0
    let l = 0
    let b = 0
    for (const c of n.children.values()) {
      agg(c)
      g += c.global
      l += c.local
      b += c.bytes
    }
    for (const leaf of n.leaves) {
      g += leaf.global
      l += leaf.local
      b += leaf.stateBytes
    }
    Object.assign(n, { global: g, local: l, bytes: b })
  }
  agg(root)
  return root
}

export default function ParamTable({ params, coords }) {
  const par = useStore((s) => s.parallel)
  const prec = useStore((s) => s.precision)
  const model = useStore((s) => s.model)
  const selectedFqn = useStore((s) => s.selectedFqn)
  const selectFqn = useStore((s) => s.selectFqn)
  const [filter, setFilter] = useState('')
  const [collapsed, setCollapsed] = useState(() => new Set())

  const perParamBytes = useMemo(() => {
    const w = DTYPE_BYTES[prec.weight]
    const g = DTYPE_BYTES[prec.grad]
    const o = DTYPE_BYTES[prec.optimDtype] * OPTIMIZERS[prec.optimizer].states
    const m = prec.master && prec.weight !== 'fp32' ? 4 : 0
    return { weight: w, trainable: w + g + o + m }
  }, [prec])

  const rows = useMemo(
    () =>
      params.map((p) => {
        const levels = shardLevels(p, par)
        const pl = placeParam(p, par, coords, levels)
        const global = prod(p.shape)
        return {
          p,
          levels,
          place: pl,
          global,
          local: pl.localNumel,
          stateBytes: pl.localNumel * (p.kind === 'buffer' ? perParamBytes.weight : perParamBytes.trainable),
        }
      }),
    [params, par, coords, perParamBytes],
  )

  const tree = useMemo(() => buildTree(rows), [rows])
  const totals = useMemo(() => {
    const trainable = rows.filter((r) => r.p.kind === 'param')
    const total = trainable.reduce((a, r) => a + r.global, 0)
    const local = trainable.reduce((a, r) => a + r.local, 0)
    const experts = trainable.filter((r) => r.p.group === 'experts').reduce((a, r) => a + r.global, 0)
    const active = model.arch === 'moe' ? total - experts + (experts * model.n_activated_experts) / model.n_routed_experts : total
    const byGroup = new Map()
    for (const r of trainable) {
      const g = byGroup.get(r.p.group) ?? { global: 0, local: 0 }
      g.global += r.global
      g.local += r.local
      byGroup.set(r.p.group, g)
    }
    return { total, local, active, byGroup }
  }, [rows, model])

  const selected = rows.find((r) => r.p.fqn === selectedFqn) ?? rows.find((r) => r.p.group === 'experts') ?? rows[1]

  const toggle = (path) =>
    setCollapsed((s) => {
      const next = new Set(s)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })

  const isCollapsed = (node) => {
    // Per-expert ModuleLists start collapsed; everything else starts open.
    const defaultClosed = node.name === 'experts' && node.children.size > 4
    return collapsed.has(node.path) ? !defaultClosed : defaultClosed
  }

  const q = filter.trim().toLowerCase()
  const flat = q ? rows.filter((r) => r.p.fqn.toLowerCase().includes(q)) : null

  const leafRow = (r, depth, label) => (
    <tr
      key={r.p.fqn}
      className={`leaf ${r.p.fqn === selected?.p.fqn ? 'sel' : ''} ${r.place.present ? '' : 'absent'}`}
      onClick={() => selectFqn(r.p.fqn)}
    >
      <td className="mono name" style={{ paddingLeft: 10 + depth * 16 }}>
        {label}
        {r.p.kind === 'buffer' && <span className="badge">buffer</span>}
      </td>
      <td className="mono num">{formatShape(r.p.shape)}</td>
      <td className="mono num local">{r.place.present ? formatShape(r.place.localShape) : '—'}</td>
      <td className="mono placement">{placementLabel(r.levels)}</td>
      <td className="mono num right">{formatCount(r.global)}</td>
      <td className="mono num right">{formatCount(r.local)}</td>
      <td className="mono num right">{formatBytes(r.local * perParamBytes.weight)}</td>
      <td className="mono num right">{formatBytes(r.stateBytes)}</td>
    </tr>
  )

  const renderNode = (node, depth) => {
    const open = !isCollapsed(node)
    return (
      <Fragment key={node.path}>
        <tr className="module" onClick={() => toggle(node.path)}>
          <td className="mono name" style={{ paddingLeft: 10 + depth * 16 }}>
            <span className="caret">{open ? '▾' : '▸'}</span>
            {node.name}
            {!open && <span className="faint"> ({node.children.size + node.leaves.length})</span>}
          </td>
          <td />
          <td />
          <td />
          <td className="mono num right">{formatCount(node.global)}</td>
          <td className="mono num right">{formatCount(node.local)}</td>
          <td />
          <td className="mono num right">{formatBytes(node.bytes)}</td>
        </tr>
        {open && (
          <>
            {node.leaves.map((l) => leafRow(l, depth + 1, l.name))}
            {[...node.children.values()].map((c) => renderNode(c, depth + 1))}
          </>
        )}
      </Fragment>
    )
  }

  return (
    <div className="params-view">
      <div className="stat-row">
        <Stat label="Total params" value={formatCount(totals.total)} />
        {model.arch === 'moe' && <Stat label="Active params / token" value={formatCount(totals.active)} />}
        <Stat label={`Params on rank ${coords.rank}`} value={formatCount(totals.local)} sub={`${((totals.local / totals.total) * 100).toFixed(2)}% of model`} />
        <Stat label={`Training state on rank ${coords.rank}`} value={formatBytes(tree.bytes)} sub="weights + grads + optimizer" />
      </div>

      <div className="group-strip">
        {[...totals.byGroup.entries()].map(([g, v]) => (
          <div key={g} className="group-cell">
            <span className="stat-label">{GROUP_LABELS[g]}</span>
            <span className="mono num">{formatCount(v.global)}</span>
            <span className="mono num faint">local {formatCount(v.local)}</span>
          </div>
        ))}
      </div>

      {selected && (
        <div className="card">
          <TensorDiagram key={selected.p.fqn} p={selected.p} par={par} coords={coords} weightDtype={prec.weight} />
        </div>
      )}

      <div className="card table-card">
        <div className="table-toolbar">
          <input className="search mono" placeholder="filter fqn…  e.g. experts, wq, norm" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <span className="faint">
            Click a row to draw its shards. Bytes use {prec.weight} weights, {prec.grad} grads, {OPTIMIZERS[prec.optimizer].label} in {prec.optimDtype}.
          </span>
        </div>
        <div className="table-scroll">
          <table className="ptable">
            <thead>
              <tr>
                <th>FQN</th>
                <th>Global shape</th>
                <th>Local shape · rank {coords.rank}</th>
                <th>Placement</th>
                <th className="right">Global</th>
                <th className="right">Local</th>
                <th className="right">Local weights</th>
                <th className="right">Local W+G+O</th>
              </tr>
            </thead>
            <tbody>
              {flat
                ? flat.map((r) => leafRow(r, 0, r.p.fqn))
                : [...tree.children.values()].map((c) => renderNode(c, 0)).concat(tree.leaves.map((l) => leafRow(l, 0, l.name)))}
            </tbody>
          </table>
        </div>
        {model.tie_embeddings && <p className="table-note faint">lm_head.weight is tied to tok_embeddings.weight and not listed separately.</p>}
      </div>
    </div>
  )
}

export function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}
