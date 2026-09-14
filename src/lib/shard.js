// Per-parameter sharding. A parameter goes through up to three placement
// levels, outermost first, mirroring how DTensor specs compose:
//
//   EP   Shard(0) over the expert axis           (grouped expert tensors)
//   TP   Shard(dim) column / row / vocab parallel
//   DP   Replicate (DDP) or Shard(0) of the TP-local tensor (FSDP2 / HSDP)
//
// Chunking follows torch.chunk semantics (ceil-sized chunks, the tail may be
// smaller or empty), which is what FSDP2 uses before padding.

import { fsdpShard, rankCoords, worldSize } from './mesh.js'
import { prod } from './format.js'

export function chunkRange(n, k, i) {
  const size = Math.ceil(n / k)
  const start = Math.min(i * size, n)
  return { start, len: Math.max(0, Math.min(size, n - start)) }
}

export function tpApplies(p, par) {
  if (par.tp <= 1 || p.tp.style === 'replicate') return false
  if (p.tp.style === 'embedding' && !par.vocabParallel) return false
  if (p.expert && !par.tpExperts) return false
  return true
}

// Levels independent of rank: each has { axis, kind, dim, degree, indexOf(coords) }.
export function shardLevels(p, par) {
  const levels = []

  if (p.expert && par.ep > 1) {
    if (p.expert.grouped) {
      levels.push({ axis: 'EP', kind: 'shard', dim: 0, degree: par.ep, indexOf: (c) => c.ep })
    } else {
      const perRank = p.expert.count / par.ep
      levels.push({
        axis: 'EP',
        kind: 'place',
        degree: par.ep,
        owner: Math.floor(p.expert.index / perRank),
        indexOf: (c) => c.ep,
      })
    }
  }

  if (tpApplies(p, par)) {
    const { style, dim, heads } = p.tp
    if (style === 'kv' && heads % par.tp !== 0) {
      // Fewer KV heads than TP ranks: each rank holds one head, replicated
      // across tp / heads ranks (Megatron-style KV replication).
      const rep = par.tp / heads
      levels.push({
        axis: 'TP',
        kind: 'shard',
        dim,
        degree: heads,
        replicas: rep,
        indexOf: (c) => Math.floor(c.tp / rep),
      })
    } else {
      levels.push({ axis: 'TP', kind: 'shard', dim, degree: par.tp, indexOf: (c) => c.tp })
    }
  }

  const expertParam = !!p.expert
  const sampleShard = fsdpShard(par, rankCoords(par, 0), expertParam)
  if (par.dp > 1) {
    if (!sampleShard || p.kind === 'buffer') {
      levels.push({ axis: 'DP', kind: 'replicate', degree: expertParam ? par.dp / par.ep : par.dp })
    } else if (sampleShard.degree > 1) {
      levels.push({
        axis: par.dpStrategy === 'hsdp' ? 'HSDP' : 'FSDP',
        kind: 'shard',
        dim: 0,
        degree: sampleShard.degree,
        indexOf: (c) => fsdpShard(par, c, expertParam).index,
      })
    } else {
      levels.push({ axis: 'DP', kind: 'replicate', degree: expertParam ? par.dp / par.ep : par.dp })
    }
  }
  return levels
}

// Local placement of `p` on the rank with coordinates `c`.
export function placeParam(p, par, c, levels = shardLevels(p, par)) {
  const ranges = p.shape.map((n) => ({ start: 0, len: n }))
  let present = true
  for (const lv of levels) {
    if (lv.kind === 'place') {
      if (lv.indexOf(c) !== lv.owner) present = false
    } else if (lv.kind === 'shard') {
      const r = ranges[lv.dim]
      const ch = chunkRange(r.len, lv.degree, lv.indexOf(c))
      r.start += ch.start
      r.len = ch.len
    }
  }
  const localShape = present ? ranges.map((r) => r.len) : null
  return {
    present,
    ranges: present ? ranges : null,
    localShape,
    localNumel: present ? prod(localShape) : 0,
  }
}

// Numel before the FSDP level (what an FSDP all-gather materializes).
export function unshardedLocalNumel(p, par, c, levels = shardLevels(p, par)) {
  const pre = levels.filter((l) => l.axis !== 'FSDP' && l.axis !== 'HSDP')
  return placeParam(p, par, c, pre).localNumel
}

export function placementLabel(levels) {
  if (!levels.length) return 'Replicate (single device)'
  return levels
    .map((l) => {
      if (l.kind === 'replicate') return `${l.axis} Replicate×${l.degree}`
      if (l.kind === 'place') return `EP place→ep${l.owner}`
      const rep = l.replicas ? ` (×${l.replicas} replicas)` : ''
      return `${l.axis} Shard(${l.dim})/${l.degree}${rep}`
    })
    .join(' → ')
}

// Enumerate shard cells (for the tensor diagram). Stops descending into
// levels once the cell count would exceed `maxCells`.
export function shardCells(p, par, maxCells = 1024) {
  const levels = shardLevels(p, par).filter((l) => l.kind === 'shard')
  let depth = 0
  let count = 1
  while (depth < levels.length && count * levels[depth].degree <= maxCells) {
    count *= levels[depth].degree
    depth++
  }
  const cells = []
  const walk = (lvl, ranges, path) => {
    if (lvl === depth) {
      cells.push({ path, ranges })
      return
    }
    const L = levels[lvl]
    for (let i = 0; i < L.degree; i++) {
      const next = ranges.map((r) => ({ ...r }))
      const ch = chunkRange(next[L.dim].len, L.degree, i)
      next[L.dim].start += ch.start
      next[L.dim].len = ch.len
      walk(lvl + 1, next, [...path, i])
    }
  }
  walk(0, p.shape.map((n) => ({ start: 0, len: n })), [])
  return { levels, depth, cells }
}

// Ranks whose level indices match `path` (prefix of shard levels).
export function ranksForPath(p, par, levels, path, limit = Infinity) {
  const out = []
  const W = worldSize(par)
  const placeLv = shardLevels(p, par).find((l) => l.kind === 'place')
  for (let r = 0; r < W && out.length < limit; r++) {
    const c = rankCoords(par, r)
    if (placeLv && placeLv.indexOf(c) !== placeLv.owner) continue
    if (path.every((idx, i) => levels[i].indexOf(c) === idx)) out.push(r)
  }
  return out
}

export function countRanksForPath(p, par, levels, path) {
  const W = worldSize(par)
  const placeLv = shardLevels(p, par).find((l) => l.kind === 'place')
  let n = W / (placeLv ? placeLv.degree : 1)
  for (let i = 0; i < path.length; i++) n /= levels[i].degree
  return n
}
