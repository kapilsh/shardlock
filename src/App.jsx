import { useStore } from './store/store.js'
import { useDerived } from './store/derived.js'
import ModelPanel from './components/ModelPanel.jsx'
import ParallelPanel from './components/ParallelPanel.jsx'
import MeshView from './components/MeshView.jsx'
import ParamTable from './components/ParamTable.jsx'
import LayerView from './components/LayerView.jsx'
import MemoryView from './components/MemoryView.jsx'
import ActivationsView from './components/ActivationsView.jsx'
import CommsView from './components/CommsView.jsx'
import './App.css'

const TABS = {
  params: 'Parameters',
  layer: 'Layer flow',
  memory: 'Memory',
  activations: 'Activations',
  comms: 'Communication',
}

const MAX_WORLD = 65536

export default function App() {
  const tab = useStore((s) => s.tab)
  const setTab = useStore((s) => s.setTab)
  const reset = useStore((s) => s.reset)
  const model = useStore((s) => s.model)
  const { params, world, coords, errors, warnings, valid } = useDerived()
  const tooBig = world > MAX_WORLD
  const ok = valid && !tooBig
  const isMoe = model.arch === 'moe'

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">{'⊞'}</span>
          <span className="brand-name">
            Shard<span className="brand-accent">lock</span>
          </span>
        </div>
        <span className="brand-tag mono">where did my tensors go? · DDP · FSDP · HSDP · TP · EP · CP</span>
        <button type="button" className="btn-ghost" onClick={reset}>
          Reset
        </button>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <ModelPanel />
          <ParallelPanel />
        </aside>

        <main className="main">
          {(errors.length > 0 || tooBig) && (
            <div className="notice error">
              {tooBig && <div>World size {world.toLocaleString('en-US')} is above the {MAX_WORLD.toLocaleString('en-US')}-rank limit of this tool.</div>}
              {errors.map((e) => (
                <div key={e}>✗ {e}</div>
              ))}
            </div>
          )}
          {warnings.length > 0 && (
            <div className="notice warn">
              {warnings.map((w) => (
                <div key={w}>⚠ {w}</div>
              ))}
            </div>
          )}

          {ok && (
            <>
              <MeshView world={world} coords={coords} isMoe={isMoe} />
              <nav className="tabs" role="tablist">
                {Object.entries(TABS).map(([k, v]) => (
                  <button key={k} type="button" role="tab" aria-selected={tab === k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>
                    {v}
                  </button>
                ))}
              </nav>
              {tab === 'params' && <ParamTable params={params} coords={coords} />}
              {tab === 'layer' && <LayerView params={params} coords={coords} />}
              {tab === 'memory' && <MemoryView params={params} coords={coords} />}
              {tab === 'activations' && <ActivationsView coords={coords} />}
              {tab === 'comms' && <CommsView params={params} coords={coords} />}
            </>
          )}
        </main>
      </div>
    </div>
  )
}
