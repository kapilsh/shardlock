import { create } from 'zustand'
import { DEFAULT_PRESET, PRESETS } from '../lib/presets.js'
import { DEFAULT_PARALLEL, worldSize } from '../lib/mesh.js'
import { DEFAULT_PRECISION } from '../lib/memory.js'

const STORAGE_KEY = 'mpv-state-v1'

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

const saved = load()

const initial = {
  presetKey: DEFAULT_PRESET,
  model: { ...PRESETS[DEFAULT_PRESET].config },
  parallel: { ...DEFAULT_PARALLEL },
  precision: { ...DEFAULT_PRECISION },
  train: { micro_batch: 1, seq_len: 4096 },
  rank: 0,
  lens: 'tp',
  tab: 'params',
  selectedFqn: null,
}

const clampRank = (rank, parallel) => Math.min(rank, Math.max(0, worldSize(parallel) - 1))

export const useStore = create((set) => ({
  ...initial,
  ...(saved
    ? {
        ...saved,
        model: { ...initial.model, ...saved.model },
        parallel: { ...initial.parallel, ...saved.parallel },
        precision: { ...initial.precision, ...saved.precision },
        train: { ...initial.train, ...saved.train },
      }
    : {}),

  applyPreset: (presetKey) =>
    set({ presetKey, model: { ...PRESETS[presetKey].config }, selectedFqn: null }),
  setModel: (patch) => set((s) => ({ model: { ...s.model, ...patch }, presetKey: 'custom' })),
  setParallel: (patch) =>
    set((s) => {
      const parallel = { ...s.parallel, ...patch }
      return { parallel, rank: clampRank(s.rank, parallel) }
    }),
  setPrecision: (patch) => set((s) => ({ precision: { ...s.precision, ...patch } })),
  setTrain: (patch) => set((s) => ({ train: { ...s.train, ...patch } })),
  setRank: (rank) => set({ rank }),
  setLens: (lens) => set({ lens }),
  setTab: (tab) => set({ tab }),
  selectFqn: (selectedFqn) => set({ selectedFqn }),
  reset: () => set({ ...initial }),
}))

useStore.subscribe((s) => {
  try {
    const { presetKey, model, parallel, precision, train, rank, lens, tab, selectedFqn } = s
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ presetKey, model, parallel, precision, train, rank, lens, tab, selectedFqn }),
    )
  } catch {
    // storage unavailable — state just won't persist
  }
})
