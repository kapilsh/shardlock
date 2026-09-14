import { useMemo, useState } from 'react'
import { useStore } from '../store/store.js'
import { formatBytes, formatCount } from '../lib/format.js'
import { validateParallel } from '../lib/mesh.js'
import { rankMemory, prepare, unitBreakdown, worldMemory } from '../lib/memory.js'
import { Stat } from './ParamTable.jsx'

const SEGMENTS = [
  { key: 'params', label: 'Weights', color: 'var(--mem-params)' },
  { key: 'grads', label: 'Grads', color: 'var(--mem-grads)' },
  { key: 'optim', label: 'Optimizer', color: 'var(--mem-optim)' },
  { key: 'master', label: 'Master fp32', color: 'var(--mem-master)' },
  { key: 'allGather', label: 'FSDP all-gather', color: 'var(--mem-gather)' },
]

export default function MemoryView({ params, coords }) {
  const par = useStore((s) => s.parallel)
  const prec = useStore((s) => s.precision)
  const model = useStore((s) => s.model)
  const [hover, setHover] = useState(null)

  const mine = useMemo(() => rankMemory(prepare(params, par), par, prec, coords.rank), [params, par, prec, coords.rank])
  const world = useMemo(() => worldMemory(params, par, prec), [params, par, prec])
  const units = useMemo(() => unitBreakdown(params, par, prec, coords.rank), [params, par, prec, coords.rank])

  const scenarios = useMemo(() => {
    const out = [{ name: 'Single GPU', par: { ...par, dp: 1, tp: 1, cp: 1, ep: 1, dpStrategy: 'ddp' } }]
    for (const s of ['ddp', 'fsdp', 'hsdp']) {
      out.push({ name: `${s.toUpperCase()} · current degrees`, par: { ...par, dpStrategy: s }, current: s === par.dpStrategy })
    }
    return out.map((sc) => {
      const bad = validateParallel(sc.par, model).errors
      if (bad.length) return { ...sc, invalid: bad[0] }
      return { ...sc, mem: rankMemory(prepare(params, sc.par), sc.par, prec, 0) }
    })
  }, [params, par, prec, model])

  const maxPeak = Math.max(...scenarios.filter((s) => s.mem).map((s) => s.mem.peak))
  const uneven = world.maxPeak !== world.minPeak

  return (
    <div className="memory-view">
      <div className="stat-row">
        <Stat label={`Peak static · rank ${coords.rank}`} value={formatBytes(mine.peak)} sub="excludes activations" />
        <Stat label="Weights" value={formatBytes(mine.params)} sub={`${formatCount(mine.localParams)} params`} />
        <Stat label="Grads" value={formatBytes(mine.grads)} />
        <Stat label="Optimizer" value={formatBytes(mine.optim + mine.master)} />
        {par.dpStrategy !== 'ddp' && <Stat label="All-gather transient" value={formatBytes(mine.allGather)} sub={par.reshardAfterForward ? 'largest FSDP unit' : 'all units (no reshard)'} />}
      </div>

      {uneven && (
        <div className="notice warn">
          Memory differs across ranks: {formatBytes(world.minPeak)} – {formatBytes(world.maxPeak)} (uneven chunks or expert placement)
          {world.sampled ? ', sampled ranks' : ''}.
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <div>
            <h2>Per-rank memory by DP strategy</h2>
            <p className="card-sub">Rank 0, same TP / CP / EP degrees. Hover a segment for its size.</p>
          </div>
          <div className="legend">
            {SEGMENTS.map((s) => (
              <span key={s.key} className="legend-item">
                <span className="swatch" style={{ background: s.color }} />
                {s.label}
              </span>
            ))}
          </div>
        </div>
        <div className="bars" onMouseLeave={() => setHover(null)}>
          {scenarios.map((sc) => (
            <div key={sc.name} className={`bar-row ${sc.current ? 'current' : ''}`}>
              <div className="bar-label">
                {sc.name}
                {sc.current && <span className="badge">current</span>}
              </div>
              <div className="bar-track">
                {sc.invalid ? (
                  <span className="faint bar-invalid">n/a — {sc.invalid}</span>
                ) : (
                  <>
                    <div className="bar-fill">
                      {SEGMENTS.filter((s) => sc.mem[s.key] > 0).map((s) => (
                        <div
                          key={s.key}
                          className="bar-seg"
                          style={{ width: `${(sc.mem[s.key] / maxPeak) * 100}%`, background: s.color }}
                          onMouseEnter={() => setHover({ sc: sc.name, seg: s.label, bytes: sc.mem[s.key] })}
                        />
                      ))}
                    </div>
                    <span className="bar-value mono num">{formatBytes(sc.mem.peak)}</span>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="bar-tooltip mono">
          {hover ? `${hover.sc} · ${hover.seg}: ${formatBytes(hover.bytes)}` : ' '}
        </div>
      </div>

      {par.dpStrategy !== 'ddp' && (
        <div className="card table-card">
          <div className="card-head">
            <div>
              <h2>FSDP units on rank {coords.rank}</h2>
              <p className="card-sub">Each unit is all-gathered (in {prec.compute}) around its forward and backward.</p>
            </div>
          </div>
          <div className="table-scroll">
            <table className="ptable">
              <thead>
                <tr>
                  <th>Unit</th>
                  <th className="right">Sharded params</th>
                  <th className="right">Gathered params</th>
                  {model.arch === 'moe' && <th className="right">Sharded experts</th>}
                  {model.arch === 'moe' && <th className="right">Gathered experts</th>}
                  <th className="right">All-gather buffer</th>
                </tr>
              </thead>
              <tbody>
                {units.map((u) => (
                  <tr key={u.unit}>
                    <td className="mono">{u.unit}</td>
                    <td className="mono num right">{formatCount(u.sharded)}</td>
                    <td className="mono num right">{formatCount(u.unsharded)}</td>
                    {model.arch === 'moe' && <td className="mono num right">{formatCount(u.expertSharded)}</td>}
                    {model.arch === 'moe' && <td className="mono num right">{formatCount(u.expertUnsharded)}</td>}
                    <td className="mono num right">{formatBytes(u.gatherBytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
