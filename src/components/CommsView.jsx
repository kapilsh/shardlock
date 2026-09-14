import { useMemo } from 'react'
import { useStore } from '../store/store.js'
import { formatBytes } from '../lib/format.js'
import { stepCollectives } from '../lib/comms.js'
import { AXIS_COLOR } from '../lib/colors.js'
import { Stat } from './ParamTable.jsx'

const ORDER = ['TP', 'CP', 'EP', 'FSDP', 'HSDP', 'DP']

export default function CommsView({ params, coords }) {
  const model = useStore((s) => s.model)
  const par = useStore((s) => s.parallel)
  const train = useStore((s) => s.train)
  const prec = useStore((s) => s.precision)
  const rows = useMemo(() => stepCollectives(model, par, train, prec, params), [model, par, train, prec, params])

  const byAxis = ORDER.map((axis) => {
    const rs = rows.filter((r) => r.axis === axis)
    return { axis, rows: rs, total: rs.reduce((a, r) => a + r.total, 0), calls: rs.reduce((a, r) => a + r.count, 0) }
  }).filter((g) => g.rows.length)

  return (
    <div className="comms-view">
      <div className="notice">
        Collectives issued by rank {coords.rank} in one training step (one micro-batch). Sizes are logical buffer sizes; on-the-wire cost depends on the
        algorithm (ring all-reduce ≈ 2× the buffer, all-gather / reduce-scatter ≈ 1×).
      </div>
      {byAxis.length === 0 ? (
        <div className="card empty">Single device — no collectives.</div>
      ) : (
        <div className="stat-row">
          {byAxis.map((g) => (
            <Stat key={g.axis} label={`${g.axis} · ${g.calls} calls`} value={formatBytes(g.total)} />
          ))}
        </div>
      )}
      {byAxis.map((g) => (
        <div key={g.axis} className="card table-card">
          <div className="card-head">
            <h2>
              <span className="swatch" style={{ background: AXIS_COLOR[g.axis] }} /> {g.axis}
            </h2>
          </div>
          <div className="table-scroll">
            <table className="ptable">
              <thead>
                <tr>
                  <th>Phase</th>
                  <th>Op</th>
                  <th>Group</th>
                  <th>What</th>
                  <th className="right">Buffer</th>
                  <th className="right">Calls</th>
                  <th className="right">Volume</th>
                </tr>
              </thead>
              <tbody>
                {g.rows.map((r, i) => (
                  <tr key={i}>
                    <td className="mono">{r.phase}</td>
                    <td className="mono">{r.op}</td>
                    <td className="mono">{r.group}</td>
                    <td>{r.what}</td>
                    <td className="mono num right">{formatBytes(r.bytes)}</td>
                    <td className="mono num right">{r.count}</td>
                    <td className="mono num right">{formatBytes(r.total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  )
}
