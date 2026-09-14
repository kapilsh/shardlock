import { useMemo } from 'react'
import { useStore } from '../store/store.js'
import { formatBytes, formatShape } from '../lib/format.js'
import { activationSections, seqDims } from '../lib/activations.js'

export default function ActivationsView({ coords }) {
  const model = useStore((s) => s.model)
  const par = useStore((s) => s.parallel)
  const train = useStore((s) => s.train)
  const prec = useStore((s) => s.precision)
  const sections = useMemo(() => activationSections(model, par, train, prec), [model, par, train, prec])
  const { sCp, sSp, sp } = seqDims(par, train)

  return (
    <div className="acts-view">
      <div className="notice">
        Forward tensor shapes on rank {coords.rank} for batch {train.micro_batch} × seq {train.seq_len.toLocaleString('en-US')} in {prec.compute}. Sequence
        axis: <code>{train.seq_len}</code> global
        {par.cp > 1 && (
          <>
            {' '}→ <code>{sCp}</code> per CP rank
          </>
        )}
        {sp && (
          <>
            {' '}→ <code>{sSp}</code> in SP regions
          </>
        )}
        . This is a shape walkthrough, not a full activation-memory estimate (no checkpointing / saved-tensor accounting).
      </div>
      {sections.map((s) => (
        <div key={s.title} className="card table-card">
          <div className="card-head">
            <h2 className="mono">{s.title}</h2>
          </div>
          <div className="table-scroll">
            <table className="ptable">
              <thead>
                <tr>
                  <th>Tensor</th>
                  <th>Local shape</th>
                  <th className="right">Bytes</th>
                  <th>Why</th>
                </tr>
              </thead>
              <tbody>
                {s.rows.map((r, i) => (
                  <tr key={i}>
                    <td className="mono">{r.name}</td>
                    <td className="mono num local">{formatShape(r.shape)}</td>
                    <td className="mono num right">{formatBytes(r.bytes)}</td>
                    <td className="faint">{r.note}</td>
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
