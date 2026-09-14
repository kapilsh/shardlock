import { useMemo } from 'react'
import { useStore } from '../store/store.js'
import { GROUPS, rankCoords } from '../lib/mesh.js'

const MAX_ROWS = 64

const LENS_COLOR = {
  tp: 'var(--ax-tp)',
  cp: 'var(--ax-cp)',
  dp: 'var(--ax-dp)',
  ep: 'var(--ax-ep)',
  edp: 'var(--ax-dp)',
  shard: 'var(--ax-hsdp)',
}

export default function MeshView({ world, coords, isMoe }) {
  const par = useStore((s) => s.parallel)
  const lens = useStore((s) => s.lens)
  const setLens = useStore((s) => s.setLens)
  const setRank = useStore((s) => s.setRank)

  const lenses = Object.entries(GROUPS).filter(([k]) => {
    if (k === 'ep' || k === 'edp') return isMoe && par.ep > 1
    if (k === 'shard') return par.dpStrategy !== 'ddp'
    return true
  })
  const activeLens = lenses.some(([k]) => k === lens) ? lens : 'tp'
  const color = LENS_COLOR[activeLens]

  const rows = useMemo(() => {
    const out = []
    const nRows = Math.min(par.dp, MAX_ROWS)
    for (let dp = 0; dp < nRows; dp++) {
      const cells = []
      for (let cp = 0; cp < par.cp; cp++) {
        const block = []
        for (let tp = 0; tp < par.tp; tp++) block.push(rankCoords(par, (dp * par.cp + cp) * par.tp + tp))
        cells.push(block)
      }
      out.push({ dp, cells })
    }
    return out
  }, [par])

  const cellsPerRow = par.cp * par.tp
  const size = cellsPerRow > 48 ? 8 : cellsPerRow > 24 ? 12 : cellsPerRow > 12 ? 16 : 22
  const test = GROUPS[activeLens].test
  const groupSize = useMemo(() => {
    let n = 0
    for (let r = 0; r < world; r++) if (test(rankCoords(par, r), coords, par)) n++
    return n
  }, [par, world, coords, test])

  return (
    <div className="card mesh-card">
      <div className="card-head">
        <div>
          <h2>Device mesh</h2>
          <p className="card-sub">
            Click a rank to inspect its local shards. Rows are DP ranks; each row holds {par.cp > 1 ? `${par.cp} CP × ` : ''}
            {par.tp} TP ranks.
          </p>
        </div>
        <div className="lens-row" role="radiogroup" aria-label="Highlight group">
          {lenses.map(([k, g]) => (
            <button key={k} type="button" className={`chip ${activeLens === k ? 'on' : ''}`} onClick={() => setLens(k)}>
              <span className="swatch" style={{ background: LENS_COLOR[k] }} />
              {g.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mesh-body">
        <div className="mesh-scroll">
          <div className="mesh">
            {rows.map(({ dp, cells }) => {
              const c0 = cells[0][0]
              const epBreak = isMoe && par.ep > 1 && dp > 0 && dp % par.ep === 0
              const hsdpBreak = par.dpStrategy === 'hsdp' && dp > 0 && dp % par.hsdpShard === 0
              return (
                <div key={dp} className={`mesh-row ${epBreak ? 'ep-break' : ''} ${hsdpBreak ? 'hsdp-break' : ''}`}>
                  <span className="mesh-row-label mono">
                    dp{dp}
                    {isMoe && par.ep > 1 && <span className="faint"> ·ep{c0.ep}</span>}
                  </span>
                  {cells.map((block, cp) => (
                    <div key={cp} className="mesh-block">
                      {block.map((c) => {
                        const inGroup = test(c, coords, par)
                        const sel = c.rank === coords.rank
                        return (
                          <button
                            key={c.rank}
                            type="button"
                            className={`rank ${sel ? 'sel' : ''}`}
                            style={{ width: size, height: size, background: inGroup ? color : undefined, opacity: inGroup || sel ? 1 : 0.9 }}
                            title={`rank ${c.rank} · dp${c.dp} cp${c.cp} tp${c.tp}${isMoe && par.ep > 1 ? ` · ep${c.ep} edp${c.edp}` : ''}`}
                            onClick={() => setRank(c.rank)}
                          >
                            {size >= 22 ? <span className="rank-num">{c.rank}</span> : null}
                          </button>
                        )
                      })}
                    </div>
                  ))}
                </div>
              )
            })}
            {par.dp > MAX_ROWS && <div className="faint mono mesh-more">… {par.dp - MAX_ROWS} more DP rows not drawn</div>}
          </div>
        </div>

        <dl className="rank-info mono">
          <div>
            <dt>rank</dt>
            <dd className="accent">{coords.rank}</dd>
          </div>
          <div>
            <dt>dp</dt>
            <dd>{coords.dp}</dd>
          </div>
          <div>
            <dt>cp</dt>
            <dd>{coords.cp}</dd>
          </div>
          <div>
            <dt>tp</dt>
            <dd>{coords.tp}</dd>
          </div>
          {isMoe && par.ep > 1 && (
            <>
              <div>
                <dt>ep</dt>
                <dd>{coords.ep}</dd>
              </div>
              <div>
                <dt>edp</dt>
                <dd>{coords.edp}</dd>
              </div>
            </>
          )}
          {par.dpStrategy === 'hsdp' && (
            <div>
              <dt>hsdp blk</dt>
              <dd>{coords.hsdpBlock}</dd>
            </div>
          )}
          <div className="rank-info-group">
            <dt>{GROUPS[activeLens].label}</dt>
            <dd>{groupSize} ranks</dd>
          </div>
        </dl>
      </div>
    </div>
  )
}
