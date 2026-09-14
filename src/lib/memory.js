// Static (non-activation) training memory per rank.

import { DTYPE_BYTES } from './format.js'
import { rankCoords, worldSize } from './mesh.js'
import { placeParam, shardLevels, unshardedLocalNumel } from './shard.js'

export const OPTIMIZERS = {
  adamw: { label: 'AdamW (2 states)', states: 2 },
  sgdm: { label: 'SGD + momentum (1 state)', states: 1 },
  none: { label: 'None (inference)', states: 0 },
}

export const DEFAULT_PRECISION = {
  weight: 'fp32', // sharded / replicated parameter storage
  compute: 'bf16', // FSDP all-gather + forward compute dtype
  grad: 'fp32',
  optimizer: 'adamw',
  optimDtype: 'fp32',
  master: false, // separate fp32 master copy (when weight dtype is not fp32)
}

export function prepare(params, par) {
  return params.map((p) => ({ p, levels: shardLevels(p, par) }))
}

export function rankMemory(prepared, par, prec, rank) {
  const c = rankCoords(par, rank)
  const wB = DTYPE_BYTES[prec.weight]
  const gB = DTYPE_BYTES[prec.grad]
  const oB = DTYPE_BYTES[prec.optimDtype]
  const cB = DTYPE_BYTES[prec.compute]
  const states = OPTIMIZERS[prec.optimizer].states
  const sharded = par.dpStrategy !== 'ddp'

  const m = { params: 0, grads: 0, master: 0, optim: 0, buffers: 0, localParams: 0, allGather: 0 }
  const units = new Map()
  for (const { p, levels } of prepared) {
    const { localNumel } = placeParam(p, par, c, levels)
    if (p.kind === 'buffer') {
      m.buffers += localNumel * wB
      continue
    }
    m.localParams += localNumel
    m.params += localNumel * wB
    m.grads += localNumel * gB
    if (prec.master && prec.weight !== 'fp32') m.master += localNumel * 4
    m.optim += localNumel * states * oB
    if (sharded) {
      const full = unshardedLocalNumel(p, par, c, levels)
      units.set(p.unit, (units.get(p.unit) ?? 0) + full * cB)
    }
  }
  if (sharded && units.size) {
    const vals = [...units.values()]
    m.allGather = par.reshardAfterForward ? Math.max(...vals) : vals.reduce((a, b) => a + b, 0)
  }
  m.static = m.params + m.grads + m.master + m.optim + m.buffers
  m.peak = m.static + m.allGather
  return m
}

// Memory for every rank (sampled when the world is huge), plus the spread.
export function worldMemory(params, par, prec, maxRanks = 512) {
  const prepared = prepare(params, par)
  const W = worldSize(par)
  const step = Math.max(1, Math.floor(W / maxRanks))
  const rows = []
  for (let r = 0; r < W; r += step) rows.push({ rank: r, ...rankMemory(prepared, par, prec, r) })
  const peaks = rows.map((x) => x.peak)
  return { rows, minPeak: Math.min(...peaks), maxPeak: Math.max(...peaks), sampled: step > 1 }
}

export function unitBreakdown(params, par, prec, rank) {
  const c = rankCoords(par, rank)
  const cB = DTYPE_BYTES[prec.compute]
  const units = new Map()
  for (const p of params) {
    if (p.kind === 'buffer') continue
    const levels = shardLevels(p, par)
    const u = units.get(p.unit) ?? { unit: p.unit, sharded: 0, unsharded: 0, expertSharded: 0, expertUnsharded: 0 }
    const local = placeParam(p, par, c, levels).localNumel
    const full = unshardedLocalNumel(p, par, c, levels)
    if (p.expert) {
      u.expertSharded += local
      u.expertUnsharded += full
    } else {
      u.sharded += local
      u.unsharded += full
    }
    units.set(p.unit, u)
  }
  return [...units.values()].map((u) => ({ ...u, gatherBytes: (u.unsharded + u.expertUnsharded) * cB }))
}
