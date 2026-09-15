// Builds the flat list of named parameters / buffers for a 2-layer (by default)
// pre-norm decoder, following the naming of the `moe_model` reference module:
//
//   tok_embeddings.weight
//   layers.{i}.attn_norm.weight
//   layers.{i}.attn.{wq,wk,wv,wo}.weight            (MHA / GQA)
//   layers.{i}.attn.{q_a_proj,q_a_norm,q_b_proj,kv_a_proj,kv_a_norm,kv_b_proj,wo}  (MLA)
//   layers.{i}.attn.{q_proj,k_proj,...,o_proj}         (KDA / gated DeltaNet linear attention)
//   layers.{i}.attn.{wq_a,wq_b,wkv,wo_a,wo_b,compressor,indexer}                  (DeepSeek-V4)
//   layers.{i}.hc_attn / hc_ffn.{fn,base,scale}      (DeepSeek-V4 hyper-connections)
//   layers.{i}.mlp_norm.weight / layers.{i}.mlp.{w_gate,w_up,w_down}.weight       (dense)
//   layers.{i}.moe_norm.weight / layers.{i}.moe.{router,experts,shared_experts}   (MoE)
//   final_norm.weight
//   lm_head.weight
//
// Each entry carries the metadata the sharding engine needs:
//   tp:     how tensor parallelism splits it ({ style, dim })
//             replicate | colwise (out-features, dim 0) | rowwise (in-features)
//             embedding (vocab rows, dim 0) | kv (colwise, may replicate heads)
//   expert: null | { grouped: true } (dim 0 is the expert axis) | { index: e }
//   unit:   the FSDP wrapping unit ('root' or 'layers.{i}')

const REPLICATE = { style: 'replicate' }
const COL = (dim = 0) => ({ style: 'colwise', dim })
const ROW = (dim = 1) => ({ style: 'rowwise', dim })

export function isMoeLayer(cfg, i) {
  // First-k dense layers (DeepSeek / GLM), then every `moe_layer_step`-th
  // layer is MoE (Llama 4 Maverick interleaves dense and MoE with step 2).
  return cfg.arch === 'moe' && i >= cfg.n_dense_layers && (i + 1) % (cfg.moe_layer_step || 1) === 0
}

// DSA lightning indexer (DeepSeek-V3.2, GLM-5.x). GLM-5 keeps a full indexer on
// the first `index_full_first` layers and then on every `index_freq`-th layer;
// the layers in between reuse the previous indexer's top-k.
export function hasIndexer(cfg, i) {
  if (cfg.attn_type !== 'mla' || !(cfg.index_n_heads > 0)) return false
  const first = cfg.index_full_first ?? 0
  const freq = cfg.index_freq || 1
  return i < first || (i - first + 1) % freq === 0
}

// Token mixer per layer. Hybrid models (Kimi K3 KDA, Qwen3-Next / Qwen3.8 gated
// DeltaNet) keep full attention on every `full_attn_period`-th layer (1-based),
// optionally on the last layer too, and use linear attention elsewhere.
// `mtp_layer_index` marks an extra MTP layer appended when counting full models.
export function isFullAttnLayer(cfg, i) {
  if (!cfg.linear_attn || cfg.linear_attn === 'none') return true
  if (cfg.mtp_layer_index === i) return cfg.mtp_full_attn ?? true
  const last = (cfg.mtp_layer_index ?? cfg.n_layers) - 1
  return (i + 1) % (cfg.full_attn_period || 1) === 0 || (cfg.full_attn_last && i === last)
}

export function layerMixer(cfg, i) {
  if (cfg.attn_type === 'dsv4') return 'dsv4'
  return isFullAttnLayer(cfg, i) ? cfg.attn_type : cfg.linear_attn
}

// DeepSeek-V4: per-layer KV compression ratio (0 = sliding window only,
// 4 = compressed + lightning indexer, 128 = heavily compressed).
export function compressRatio(cfg, i) {
  return cfg.compress_ratios?.[i] ?? 0
}

// DeepSeek-V4 routes the first `n_hash_layers` layers by a token-id lookup table.
export function isHashLayer(cfg, i) {
  return cfg.attn_type === 'dsv4' && i < (cfg.n_hash_layers ?? 0)
}

export function buildModel(cfg) {
  const params = []
  const add = (fqn, shape, meta) => {
    params.push({
      fqn,
      shape,
      kind: 'param',
      trainable: true,
      tp: REPLICATE,
      expert: null,
      layer: null,
      unit: 'root',
      ...meta,
    })
  }

  const D = cfg.dim
  const V = cfg.vocab_size

  add('tok_embeddings.weight', [V, D], {
    group: 'embedding',
    tp: { style: 'embedding', dim: 0 },
  })

  const dsv4 = cfg.attn_type === 'dsv4'
  const hyperConn = (P, lm) => {
    // Hyper-connections: mix `hc_mult` residual copies (Sinkhorn-normalized).
    const m = cfg.hc_mult
    const g = { ...lm, group: 'hyperconn' }
    add(`${P}.fn`, [(2 + m) * m, m * D], g)
    add(`${P}.base`, [(2 + m) * m], g)
    add(`${P}.scale`, [3], g)
  }
  const attnRes = (P, lm) => {
    add(`${P}_res_norm.weight`, [D], { ...lm, group: 'attn_res' })
    add(`${P}_res_proj.weight`, [1, D], { ...lm, group: 'attn_res' })
  }

  for (let i = 0; i < cfg.n_layers; i++) {
    const L = `layers.${i}`
    const lm = { layer: i, unit: L }

    if (dsv4) hyperConn(`${L}.hc_attn`, lm)
    if (cfg.attn_res) attnRes(`${L}.attn`, lm)
    add(`${L}.attn_norm.weight`, [D], { ...lm, group: 'norm' })
    const mixer = layerMixer(cfg, i)
    if (mixer === 'kda') addKda(cfg, add, `${L}.attn`, lm)
    else if (mixer === 'gdn') addGatedDeltaNet(cfg, add, `${L}.attn`, lm)
    else if (mixer === 'dsv4') addDsv4Attention(cfg, add, `${L}.attn`, lm)
    else addAttention(cfg, add, `${L}.attn`, lm)

    if (dsv4) hyperConn(`${L}.hc_ffn`, lm)
    if (cfg.attn_res) attnRes(`${L}.mlp`, lm)
    if (isMoeLayer(cfg, i)) {
      add(`${L}.moe_norm.weight`, [D], { ...lm, group: 'norm' })
      addMoe(cfg, add, `${L}.moe`, lm)
    } else {
      add(`${L}.mlp_norm.weight`, [D], { ...lm, group: 'norm' })
      addSwiGLU(add, `${L}.mlp`, D, cfg.ffn_dim, cfg.mlp_bias, { ...lm, group: 'ffn' })
    }
  }

  if (cfg.attn_res) {
    add('output_res_norm.weight', [D], { group: 'attn_res' })
    add('output_res_proj.weight', [1, D], { group: 'attn_res' })
  }
  if (dsv4) {
    const m = cfg.hc_mult
    add('hc_head.fn', [m, m * D], { group: 'hyperconn' })
    add('hc_head.base', [m], { group: 'hyperconn' })
    add('hc_head.scale', [1], { group: 'hyperconn' })
  }
  add('final_norm.weight', [D], { group: 'norm' })
  if (!cfg.tie_embeddings) {
    add('lm_head.weight', [V, D], { group: 'lm_head', tp: { style: 'embedding', dim: 0 } })
  }

  return params
}

function addAttention(cfg, add, P, lm) {
  const D = cfg.dim
  const H = cfg.n_heads
  const g = { ...lm, group: 'attention' }

  if (cfg.attn_type === 'mla') {
    const qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    if (cfg.q_lora_rank > 0) {
      add(`${P}.q_a_proj.weight`, [cfg.q_lora_rank, D], g)
      add(`${P}.q_a_norm.weight`, [cfg.q_lora_rank], { ...g, group: 'norm' })
      add(`${P}.q_b_proj.weight`, [H * qk, cfg.q_lora_rank], { ...g, tp: COL() })
    } else {
      add(`${P}.wq.weight`, [H * qk, D], { ...g, tp: COL() })
    }
    if (hasIndexer(cfg, lm.layer)) {
      // Replicated over TP (plain Linear in the reference implementations).
      const ih = cfg.index_n_heads
      const id = cfg.index_head_dim
      add(`${P}.indexer.wq_b.weight`, [ih * id, cfg.q_lora_rank > 0 ? cfg.q_lora_rank : D], g)
      add(`${P}.indexer.wk.weight`, [id, D], g)
      add(`${P}.indexer.k_norm.weight`, [id], { ...g, group: 'norm' })
      add(`${P}.indexer.k_norm.bias`, [id], { ...g, group: 'norm' })
      add(`${P}.indexer.weights_proj.weight`, [ih, D], g)
    }
    add(`${P}.kv_a_proj.weight`, [cfg.kv_lora_rank + cfg.qk_rope_head_dim, D], g)
    add(`${P}.kv_a_norm.weight`, [cfg.kv_lora_rank], { ...g, group: 'norm' })
    add(`${P}.kv_b_proj.weight`, [H * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank], {
      ...g,
      tp: COL(),
    })
    if (cfg.attn_output_gate) add(`${P}.g_proj.weight`, [H * cfg.v_head_dim, D], { ...g, tp: COL() })
    add(`${P}.wo.weight`, [D, H * cfg.v_head_dim], { ...g, tp: ROW() })
    return
  }

  const KV = cfg.attn_type === 'mha' ? H : cfg.n_kv_heads
  const hd = cfg.head_dim
  const kvTp = { style: 'kv', dim: 0, heads: KV }

  // With an output gate (Qwen3-Next / Qwen3.8) the q projection also emits the gate.
  const qOut = H * hd * (cfg.attn_output_gate ? 2 : 1)
  add(`${P}.wq.weight`, [qOut, D], { ...g, tp: COL() })
  if (cfg.qkv_bias) add(`${P}.wq.bias`, [qOut], { ...g, tp: COL() })
  add(`${P}.wk.weight`, [KV * hd, D], { ...g, tp: kvTp })
  if (cfg.qkv_bias) add(`${P}.wk.bias`, [KV * hd], { ...g, tp: kvTp })
  add(`${P}.wv.weight`, [KV * hd, D], { ...g, tp: kvTp })
  if (cfg.qkv_bias) add(`${P}.wv.bias`, [KV * hd], { ...g, tp: kvTp })
  add(`${P}.wo.weight`, [D, H * hd], { ...g, tp: ROW() })
  if (cfg.o_bias) add(`${P}.wo.bias`, [D], g)
  if (cfg.qk_norm) {
    if (cfg.qk_norm_type === 'full') {
      // One RMSNorm over all heads (MiniMax-M2): sharded like the projection.
      add(`${P}.q_norm.weight`, [H * hd], { ...g, group: 'norm', tp: COL() })
      add(`${P}.k_norm.weight`, [KV * hd], { ...g, group: 'norm', tp: kvTp })
    } else {
      add(`${P}.q_norm.weight`, [hd], { ...g, group: 'norm' })
      add(`${P}.k_norm.weight`, [hd], { ...g, group: 'norm' })
    }
  }
  if (cfg.attn_sinks) add(`${P}.sinks`, [H], { ...g, tp: COL() })
}

// Kimi Delta Attention (Kimi K3 / Kimi Linear). Heads shard over TP like q/k/v;
// the low-rank decay input and per-head-dim tensors stay replicated.
function addKda(cfg, add, P, lm) {
  const D = cfg.dim
  const H = cfg.la_num_heads
  const hd = cfg.la_head_dim
  const K = cfg.la_conv
  const g = { ...lm, group: 'attention' }
  for (const x of ['q', 'k', 'v']) add(`${P}.${x}_proj.weight`, [H * hd, D], { ...g, tp: COL() })
  for (const x of ['q', 'k', 'v']) add(`${P}.${x}_conv1d.weight`, [H * hd, 1, K], { ...g, tp: COL() })
  add(`${P}.f_a_proj.weight`, [hd, D], g)
  add(`${P}.f_b_proj.weight`, [H * hd, hd], { ...g, tp: COL() })
  add(`${P}.A_log`, [hd], g) // the K3 checkpoint stores head_dim entries
  add(`${P}.dt_bias`, [H * hd], { ...g, tp: COL() })
  add(`${P}.b_proj.weight`, [H, D], { ...g, tp: COL() })
  add(`${P}.g_proj.weight`, [H * hd, D], { ...g, tp: COL() })
  add(`${P}.o_norm.weight`, [hd], { ...g, group: 'norm' })
  add(`${P}.o_proj.weight`, [D, H * hd], { ...g, tp: ROW() })
}

// Gated DeltaNet (Qwen3-Next / Qwen3.8). q,k,v share one projection and conv.
function addGatedDeltaNet(cfg, add, P, lm) {
  const D = cfg.dim
  const Hk = cfg.la_num_k_heads
  const dk = cfg.la_head_dim
  const Hv = cfg.la_num_heads
  const dv = cfg.la_v_head_dim
  const qkv = 2 * Hk * dk + Hv * dv
  const g = { ...lm, group: 'attention' }
  add(`${P}.in_proj_qkv.weight`, [qkv, D], { ...g, tp: COL() })
  add(`${P}.in_proj_z.weight`, [Hv * dv, D], { ...g, tp: COL() })
  add(`${P}.in_proj_a.weight`, [Hv, D], { ...g, tp: COL() })
  add(`${P}.in_proj_b.weight`, [Hv, D], { ...g, tp: COL() })
  add(`${P}.conv1d.weight`, [qkv, 1, cfg.la_conv], { ...g, tp: COL() })
  add(`${P}.A_log`, [Hv], { ...g, tp: COL() })
  add(`${P}.dt_bias`, [Hv], { ...g, tp: COL() })
  add(`${P}.norm.weight`, [dv], { ...g, group: 'norm' })
  add(`${P}.out_proj.weight`, [D, Hv * dv], { ...g, tp: ROW() })
}

// DeepSeek-V4 attention: low-rank q, single-head (MQA) window KV, grouped
// low-rank output, plus a KV compressor (and lightning indexer at ratio 4).
// TP follows the reference: wq_b / wo_a / indexer projections column-parallel,
// wo_b row-parallel, everything else replicated.
function addDsv4Attention(cfg, add, P, lm) {
  const D = cfg.dim
  const H = cfg.n_heads
  const hd = cfg.head_dim
  const qr = cfg.q_lora_rank
  const G = cfg.o_groups
  const ratio = compressRatio(cfg, lm.layer)
  const g = { ...lm, group: 'attention' }
  const norm = { ...g, group: 'norm' }
  const compressor = (C, dim, r) => {
    const w = (r === 4 ? 2 : 1) * dim // ratio 4 uses overlapping windows
    add(`${C}.ape`, [r, w], g)
    add(`${C}.wkv.weight`, [w, D], g)
    add(`${C}.wgate.weight`, [w, D], g)
    add(`${C}.norm.weight`, [dim], norm)
  }
  add(`${P}.attn_sink`, [H], { ...g, tp: COL() })
  add(`${P}.wq_a.weight`, [qr, D], g)
  add(`${P}.q_norm.weight`, [qr], norm)
  add(`${P}.wq_b.weight`, [H * hd, qr], { ...g, tp: COL() })
  add(`${P}.wkv.weight`, [hd, D], g)
  add(`${P}.kv_norm.weight`, [hd], norm)
  if (ratio) compressor(`${P}.compressor`, hd, ratio)
  if (ratio === 4 && cfg.index_n_heads > 0) {
    add(`${P}.indexer.wq_b.weight`, [cfg.index_n_heads * cfg.index_head_dim, qr], { ...g, tp: COL() })
    add(`${P}.indexer.weights_proj.weight`, [cfg.index_n_heads, D], { ...g, tp: COL() })
    compressor(`${P}.indexer.compressor`, cfg.index_head_dim, 4)
  }
  add(`${P}.wo_a.weight`, [G * cfg.o_lora_rank, (H * hd) / G], { ...g, tp: COL() })
  add(`${P}.wo_b.weight`, [D, G * cfg.o_lora_rank], { ...g, tp: ROW() })
}

function addSwiGLU(add, P, D, I, bias, meta) {
  add(`${P}.w_gate.weight`, [I, D], { ...meta, tp: COL() })
  if (bias) add(`${P}.w_gate.bias`, [I], { ...meta, tp: COL() })
  add(`${P}.w_up.weight`, [I, D], { ...meta, tp: COL() })
  if (bias) add(`${P}.w_up.bias`, [I], { ...meta, tp: COL() })
  add(`${P}.w_down.weight`, [D, I], { ...meta, tp: ROW() })
  if (bias) add(`${P}.w_down.bias`, [D], meta)
}

function addMoe(cfg, add, P, lm) {
  const D = cfg.dim
  const E = cfg.n_routed_experts
  const I = cfg.moe_inter_dim

  // Routed experts can run in a narrower latent width (Kimi K3).
  const Dx = cfg.moe_latent_dim > 0 ? cfg.moe_latent_dim : D
  const hash = isHashLayer(cfg, lm.layer)

  add(`${P}.router.weight`, [E, D], { ...lm, group: 'router' })
  if (cfg.router_bias) add(`${P}.router.bias`, [E], { ...lm, group: 'router' })
  if (hash) {
    // token id -> expert ids lookup (int64), used instead of top-k scores
    add(`${P}.router.tid2eid`, [cfg.vocab_size, cfg.n_activated_experts], { ...lm, group: 'router', kind: 'buffer', trainable: false, elemBytes: 8 })
  } else if (cfg.balance_bias) {
    add(`${P}.router.balance_bias`, [E], { ...lm, group: 'router', kind: 'buffer', trainable: false })
  }
  if (cfg.moe_latent_dim > 0) {
    const lg = { ...lm, group: 'latent' }
    add(`${P}.latent_down.weight`, [Dx, D], lg)
    add(`${P}.latent_norm.weight`, [Dx], lg)
    add(`${P}.latent_up.weight`, [D, Dx], lg)
  }

  const em = { ...lm, group: 'experts' }
  if (cfg.expert_layout === 'grouped') {
    const ex = { ...em, expert: { grouped: true } }
    add(`${P}.experts.w_gate`, [E, I, Dx], { ...ex, tp: COL(1) })
    if (cfg.expert_bias) add(`${P}.experts.w_gate_bias`, [E, I], { ...ex, tp: COL(1) })
    add(`${P}.experts.w_up`, [E, I, Dx], { ...ex, tp: COL(1) })
    if (cfg.expert_bias) add(`${P}.experts.w_up_bias`, [E, I], { ...ex, tp: COL(1) })
    add(`${P}.experts.w_down`, [E, Dx, I], { ...ex, tp: ROW(2) })
    if (cfg.expert_bias) add(`${P}.experts.w_down_bias`, [E, Dx], ex)
  } else {
    for (let e = 0; e < E; e++) {
      addSwiGLU(add, `${P}.experts.${e}`, Dx, I, cfg.expert_bias, { ...em, expert: { index: e, count: E } })
    }
  }

  if (cfg.n_shared_experts > 0) {
    addSwiGLU(add, `${P}.shared_experts`, D, I * cfg.n_shared_experts, cfg.expert_bias, {
      ...lm,
      group: 'shared_experts',
    })
    if (cfg.shared_expert_gate) add(`${P}.shared_gate.weight`, [1, D], { ...lm, group: 'shared_experts' })
  }
}

export const GROUP_LABELS = {
  embedding: 'Embedding',
  attention: 'Attention',
  norm: 'Norms',
  ffn: 'Dense FFN',
  router: 'Router',
  experts: 'Routed experts',
  shared_experts: 'Shared experts',
  latent: 'Latent MoE proj',
  hyperconn: 'Hyper-connections',
  attn_res: 'Attention residuals',
  lm_head: 'LM head',
}

export function validateModel(cfg) {
  const errors = []
  const warnings = []
  const pos = ['vocab_size', 'dim', 'n_layers', 'n_heads']
  for (const k of pos) if (!(cfg[k] >= 1)) errors.push(`${k} must be ≥ 1`)
  if (cfg.attn_type === 'mha' && cfg.n_kv_heads !== cfg.n_heads) {
    warnings.push('MHA ignores n_kv_heads (uses n_heads KV heads)')
  }
  if (cfg.attn_type === 'gqa' && cfg.n_heads % cfg.n_kv_heads !== 0) {
    errors.push('GQA requires n_heads divisible by n_kv_heads')
  }
  if (cfg.attn_type === 'dsv4') {
    if (!(cfg.o_groups >= 1) || (cfg.n_heads * cfg.head_dim) % cfg.o_groups !== 0) {
      errors.push('DeepSeek-V4 needs n_heads × head_dim divisible by o_groups')
    }
    if (!(cfg.hc_mult >= 1)) errors.push('hc_mult must be ≥ 1')
  }
  if (cfg.linear_attn === 'gdn' && cfg.la_num_heads % cfg.la_num_k_heads !== 0) {
    errors.push('gated DeltaNet needs value heads divisible by key heads')
  }
  if (cfg.arch === 'moe') {
    if (cfg.n_activated_experts > cfg.n_routed_experts) {
      errors.push('n_activated_experts cannot exceed n_routed_experts')
    }
    if (!(cfg.moe_layer_step >= 1)) errors.push('moe_layer_step must be ≥ 1')
    else if (!Array.from({ length: cfg.n_layers }, (_, i) => isMoeLayer(cfg, i)).some(Boolean)) {
      warnings.push('no MoE layer within n_layers (check first-k dense / MoE layer step)')
    }
  }
  return { errors, warnings }
}
