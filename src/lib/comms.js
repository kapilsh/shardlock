// Collectives issued by one rank during one training step (one micro-batch).
// Message size is the logical buffer size of the collective (the full output
// for all-gather / all-reduce, the input for reduce-scatter).

import { DTYPE_BYTES } from './format.js'
import { isMoeLayer } from './model.js'
import { rankCoords } from './mesh.js'
import { placeParam, shardLevels, unshardedLocalNumel } from './shard.js'
import { kvLocalHeads, moeTokens, moeTpRegion, seqDims } from './activations.js'

const DDP_BUCKET = 25 * 2 ** 20

export function stepCollectives(cfg, par, train, prec, params) {
  const rows = []
  const cB = DTYPE_BYTES[prec.compute]
  const gB = DTYPE_BYTES[prec.grad]
  const b = train.micro_batch
  const D = cfg.dim
  const { sCp, sp } = seqDims(par, train)
  const tpOn = par.tp > 1
  const push = (r) => rows.push({ count: 1, ...r, total: (r.count ?? 1) * r.bytes })
  const hidden = b * sCp * D * cB

  // ---- Tensor parallel ------------------------------------------------------
  if (tpOn) {
    const grp = `TP·${par.tp}`
    const region = (where, n) => {
      if (sp) {
        push({ axis: 'TP', phase: 'fwd', op: 'all-gather', group: grp, what: `${where}: SP → TP region`, bytes: hidden, count: n })
        push({ axis: 'TP', phase: 'fwd', op: 'reduce-scatter', group: grp, what: `${where}: TP → SP region`, bytes: hidden, count: n })
        push({ axis: 'TP', phase: 'bwd', op: 'all-gather', group: grp, what: `${where}: grad of reduce-scatter`, bytes: hidden, count: n })
        push({ axis: 'TP', phase: 'bwd', op: 'reduce-scatter', group: grp, what: `${where}: grad of all-gather`, bytes: hidden, count: n })
      } else {
        push({ axis: 'TP', phase: 'fwd', op: 'all-reduce', group: grp, what: `${where}: row-parallel output`, bytes: hidden, count: n })
        push({ axis: 'TP', phase: 'bwd', op: 'all-reduce', group: grp, what: `${where}: column-parallel input grad`, bytes: hidden, count: n })
      }
    }
    if (par.vocabParallel) {
      push({ axis: 'TP', phase: 'fwd', op: sp ? 'reduce-scatter' : 'all-reduce', group: grp, what: 'tok_embeddings (vocab-parallel lookup)', bytes: hidden })
      if (sp) push({ axis: 'TP', phase: 'bwd', op: 'all-gather', group: grp, what: 'tok_embeddings grad', bytes: hidden })
    }
    region('attention', cfg.n_layers)
    const nDense = cfg.arch === 'moe' ? Math.min(cfg.n_dense_layers, cfg.n_layers) : cfg.n_layers
    if (nDense) region('dense mlp', nDense)
    const nMoe = cfg.n_layers - nDense
    if (nMoe && moeTpRegion(cfg, par)) region('moe block', nMoe)
    if (par.vocabParallel) {
      push({ axis: 'TP', phase: sp ? 'fwd' : 'bwd', op: sp ? 'all-gather' : 'all-reduce', group: grp, what: 'lm_head input', bytes: hidden })
      if (sp) push({ axis: 'TP', phase: 'bwd', op: 'reduce-scatter', group: grp, what: 'lm_head input grad', bytes: hidden })
    }
  }

  // ---- Context parallel -----------------------------------------------------
  if (par.cp > 1) {
    const grp = `CP·${par.cp}`
    let kvNumel
    if (cfg.attn_type === 'mla') {
      const H = cfg.n_heads / par.tp
      kvNumel = b * H * sCp * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
    } else {
      kvNumel = b * kvLocalHeads(cfg, par) * sCp * cfg.head_dim * 2
    }
    const kv = kvNumel * cB
    const L = cfg.n_layers
    if (par.cpStyle === 'ring') {
      push({ axis: 'CP', phase: 'fwd', op: 'send/recv', group: grp, what: 'ring attention: K,V chunk per step', bytes: kv, count: L * (par.cp - 1) })
      push({ axis: 'CP', phase: 'bwd', op: 'send/recv', group: grp, what: 'ring attention: K,V + dK,dV per step', bytes: 2 * kv, count: L * (par.cp - 1) })
    } else {
      push({ axis: 'CP', phase: 'fwd', op: 'all-gather', group: grp, what: 'all-gather K,V (full sequence)', bytes: kv * par.cp, count: L })
      push({ axis: 'CP', phase: 'bwd', op: 'all-gather', group: grp, what: 're-gather K,V', bytes: kv * par.cp, count: L })
      push({ axis: 'CP', phase: 'bwd', op: 'reduce-scatter', group: grp, what: 'dK,dV back to owners', bytes: kv * par.cp, count: L })
    }
  }

  // ---- Expert parallel ------------------------------------------------------
  if (cfg.arch === 'moe' && par.ep > 1) {
    const grp = `EP·${par.ep}`
    const nMoe = Array.from({ length: cfg.n_layers }, (_, i) => isMoeLayer(cfg, i)).filter(Boolean).length
    if (nMoe) {
      const { dispatched } = moeTokens(cfg, par, train)
      const bytes = dispatched * D * cB
      push({ axis: 'EP', phase: 'fwd', op: 'all-to-all', group: grp, what: 'token dispatch', bytes, count: nMoe })
      push({ axis: 'EP', phase: 'fwd', op: 'all-to-all', group: grp, what: 'token combine', bytes, count: nMoe })
      push({ axis: 'EP', phase: 'bwd', op: 'all-to-all', group: grp, what: 'grad of combine', bytes, count: nMoe })
      push({ axis: 'EP', phase: 'bwd', op: 'all-to-all', group: grp, what: 'grad of dispatch', bytes, count: nMoe })
    }
  }

  // ---- Data parallel --------------------------------------------------------
  if (par.dp > 1) {
    const c = rankCoords(par, 0)
    const units = new Map()
    for (const p of params) {
      if (p.kind === 'buffer') continue
      const levels = shardLevels(p, par)
      const key = `${p.unit}${p.expert ? ' · experts' : ''}`
      const u = units.get(key) ?? { key, expert: !!p.expert, full: 0, local: 0, root: p.unit === 'root' }
      u.full += unshardedLocalNumel(p, par, c, levels)
      u.local += placeParam(p, par, c, levels).localNumel
      units.set(key, u)
    }
    const dpGroup = (expert) => (expert ? `EDP·${par.dp / par.ep}` : `DP·${par.dp}`)

    if (par.dpStrategy === 'ddp') {
      for (const expert of [false, true]) {
        const numel = [...units.values()].filter((u) => u.expert === expert).reduce((a, u) => a + u.full, 0)
        if (!numel || (expert && par.dp / par.ep === 1)) continue
        const bytes = numel * gB
        const buckets = Math.max(1, Math.ceil(bytes / DDP_BUCKET))
        push({
          axis: 'DP',
          phase: 'bwd',
          op: 'all-reduce',
          group: dpGroup(expert),
          what: `${expert ? 'expert ' : ''}gradient buckets (25 MiB)`,
          bytes: bytes / buckets,
          count: buckets,
        })
      }
    } else {
      const hsdp = par.dpStrategy === 'hsdp'
      for (const u of units.values()) {
        const shardDeg = hsdp ? par.hsdpShard / (u.expert ? par.ep : 1) : u.expert ? par.dp / par.ep : par.dp
        const replicas = hsdp ? par.dp / par.hsdpShard : 1
        const axis = hsdp ? 'HSDP' : 'FSDP'
        if (shardDeg > 1) {
          const grp = `${axis}·${shardDeg}`
          const gather = u.full * cB
          push({ axis, phase: 'fwd', op: 'all-gather', group: grp, what: `${u.key} params`, bytes: gather })
          if (par.reshardAfterForward && !u.root) {
            push({ axis, phase: 'bwd', op: 'all-gather', group: grp, what: `${u.key} params (resharded after fwd)`, bytes: gather })
          }
          push({ axis, phase: 'bwd', op: 'reduce-scatter', group: grp, what: `${u.key} grads`, bytes: u.full * gB })
        }
        if (replicas > 1) {
          push({ axis, phase: 'bwd', op: 'all-reduce', group: `replica·${replicas}`, what: `${u.key} sharded grads across HSDP blocks`, bytes: u.local * gB })
        }
      }
    }
  }
  return rows
}
