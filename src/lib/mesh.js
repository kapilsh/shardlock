// Device mesh: world = dp × cp × tp (outer → inner). TP is innermost so it stays
// inside a node (NVLink); CP next; DP outermost.
//
// Expert parallelism is carved out of the DP dimension (DeepSpeed-MoE /
// Megatron convention): the dp ranks of each (cp, tp) slice are split into
// contiguous EP groups of size `ep`. Expert params are sharded across the EP
// group and then data-parallel (replicated or FSDP-sharded) across the
// remaining dp / ep "expert data parallel" (EDP) ranks.
//
// HSDP (hybrid sharded DP, a.k.a. FSDP HYBRID_SHARD): params are FSDP-sharded
// within blocks of `hsdpShard` dp ranks and replicated across blocks.
//
// The dims list is ordered so a `pp` axis can be prepended later.

export const DP_STRATEGIES = {
  ddp: 'DDP',
  fsdp: 'FSDP',
  hsdp: 'HSDP',
}

export const DEFAULT_PARALLEL = {
  dpStrategy: 'fsdp',
  dp: 4,
  tp: 2,
  cp: 1,
  ep: 2,
  hsdpShard: 2,
  sp: true, // sequence parallel (with TP)
  vocabParallel: true, // shard tok_embeddings / lm_head over TP
  tpExperts: true, // TP-shard routed experts
  cpStyle: 'ring', // 'ring' | 'allgather'
  reshardAfterForward: true,
}

export function worldSize(par) {
  return par.dp * par.cp * par.tp
}

export function rankCoords(par, rank) {
  const tp = rank % par.tp
  const cp = Math.floor(rank / par.tp) % par.cp
  const dp = Math.floor(rank / (par.tp * par.cp))
  const ep = dp % par.ep
  const edp = Math.floor(dp / par.ep)
  const h = hsdpBlock(par)
  return {
    rank,
    dp,
    cp,
    tp,
    ep,
    edp,
    hsdpBlock: Math.floor(dp / h),
    hsdpIndex: dp % h,
  }
}

export function coordsToRank(par, { dp, cp, tp }) {
  return (dp * par.cp + cp) * par.tp + tp
}

function hsdpBlock(par) {
  return par.dpStrategy === 'hsdp' ? Math.max(1, Math.min(par.hsdpShard, par.dp)) : par.dp
}

// FSDP sharding group size + this rank's index within it.
// `expert` params shard over the EDP slice of the group.
export function fsdpShard(par, c, expert) {
  if (par.dpStrategy === 'ddp') return null
  const epDeg = expert ? par.ep : 1
  if (par.dpStrategy === 'fsdp') {
    return expert ? { degree: par.dp / par.ep, index: c.edp } : { degree: par.dp, index: c.dp }
  }
  const h = hsdpBlock(par)
  return { degree: h / epDeg, index: Math.floor(c.hsdpIndex / epDeg) }
}

// Group membership predicates, used by the mesh lens.
export const GROUPS = {
  tp: { label: 'TP group', test: (a, b) => a.dp === b.dp && a.cp === b.cp },
  cp: { label: 'CP group', test: (a, b) => a.dp === b.dp && a.tp === b.tp },
  dp: { label: 'DP group', test: (a, b) => a.cp === b.cp && a.tp === b.tp },
  ep: {
    label: 'EP group',
    test: (a, b) => a.cp === b.cp && a.tp === b.tp && a.edp === b.edp,
  },
  edp: {
    label: 'Expert-DP group',
    test: (a, b) => a.cp === b.cp && a.tp === b.tp && a.ep === b.ep,
  },
  shard: {
    label: 'FSDP shard group',
    test: (a, b, par) =>
      a.cp === b.cp &&
      a.tp === b.tp &&
      (par.dpStrategy !== 'hsdp' || a.hsdpBlock === b.hsdpBlock) &&
      par.dpStrategy !== 'ddp',
  },
}

const isPosInt = (x) => Number.isInteger(x) && x >= 1

export function validateParallel(par, cfg, train) {
  const errors = []
  const warnings = []
  for (const k of ['dp', 'tp', 'cp', 'ep']) {
    if (!isPosInt(par[k])) errors.push(`${k} must be a positive integer`)
  }
  if (errors.length) return { errors, warnings }

  const moe = cfg.arch === 'moe'
  if (par.dp % par.ep !== 0) errors.push(`dp (${par.dp}) must be divisible by ep (${par.ep})`)
  if (moe && cfg.n_routed_experts % par.ep !== 0) {
    errors.push(`n_routed_experts (${cfg.n_routed_experts}) must be divisible by ep (${par.ep})`)
  }
  if (!moe && par.ep > 1) warnings.push('EP > 1 has no effect on a dense model')

  if (par.dpStrategy === 'hsdp') {
    if (!isPosInt(par.hsdpShard)) errors.push('HSDP shard size must be a positive integer')
    else {
      if (par.dp % par.hsdpShard !== 0) {
        errors.push(`dp (${par.dp}) must be divisible by the HSDP shard size (${par.hsdpShard})`)
      }
      if (par.hsdpShard % par.ep !== 0) {
        errors.push(`HSDP shard size (${par.hsdpShard}) must be divisible by ep (${par.ep})`)
      }
      if (par.hsdpShard === par.dp) warnings.push('HSDP shard size = dp: identical to FSDP')
      if (par.hsdpShard === 1) warnings.push('HSDP shard size = 1: identical to DDP')
    }
  }

  if (par.tp > 1) {
    const H = cfg.n_heads
    if (H % par.tp !== 0) errors.push(`n_heads (${H}) must be divisible by tp (${par.tp})`)
    if (cfg.attn_type === 'gqa') {
      const KV = cfg.n_kv_heads
      if (KV % par.tp !== 0 && par.tp % KV !== 0) {
        errors.push(`n_kv_heads (${KV}) and tp (${par.tp}) must divide one another`)
      } else if (KV < par.tp) {
        warnings.push(`n_kv_heads (${KV}) < tp (${par.tp}): KV heads are replicated across TP ranks`)
      }
    }
    const uneven = (name, n) => {
      if (n % par.tp !== 0) warnings.push(`${name} (${n}) not divisible by tp: uneven (padded) shards`)
    }
    if (par.vocabParallel) uneven('vocab_size', cfg.vocab_size)
    if (!moe || cfg.n_dense_layers > 0) uneven('ffn_dim', cfg.ffn_dim)
    if (moe && par.tpExperts) uneven('moe_inter_dim', cfg.moe_inter_dim)
  } else if (par.sp) {
    warnings.push('Sequence parallel has no effect without TP')
  }

  if (train && train.seq_len % (par.cp * (par.sp ? par.tp : 1)) !== 0) {
    warnings.push('seq_len not divisible by cp (× tp with SP): uneven sequence shards')
  }
  return { errors, warnings }
}
