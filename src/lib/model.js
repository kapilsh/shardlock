// Builds the flat list of named parameters / buffers for a 2-layer (by default)
// pre-norm decoder, following the naming of the `moe_model` reference module:
//
//   tok_embeddings.weight
//   layers.{i}.attn_norm.weight
//   layers.{i}.attn.{wq,wk,wv,wo}.weight            (MHA / GQA)
//   layers.{i}.attn.{q_a_proj,q_a_norm,q_b_proj,kv_a_proj,kv_a_norm,kv_b_proj,wo}  (MLA)
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
  return cfg.arch === 'moe' && i >= cfg.n_dense_layers
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

  for (let i = 0; i < cfg.n_layers; i++) {
    const L = `layers.${i}`
    const lm = { layer: i, unit: L }

    add(`${L}.attn_norm.weight`, [D], { ...lm, group: 'norm' })
    addAttention(cfg, add, `${L}.attn`, lm)

    if (isMoeLayer(cfg, i)) {
      add(`${L}.moe_norm.weight`, [D], { ...lm, group: 'norm' })
      addMoe(cfg, add, `${L}.moe`, lm)
    } else {
      add(`${L}.mlp_norm.weight`, [D], { ...lm, group: 'norm' })
      addSwiGLU(add, `${L}.mlp`, D, cfg.ffn_dim, cfg.mlp_bias, { ...lm, group: 'ffn' })
    }
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
    add(`${P}.kv_a_proj.weight`, [cfg.kv_lora_rank + cfg.qk_rope_head_dim, D], g)
    add(`${P}.kv_a_norm.weight`, [cfg.kv_lora_rank], { ...g, group: 'norm' })
    add(`${P}.kv_b_proj.weight`, [H * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank], {
      ...g,
      tp: COL(),
    })
    add(`${P}.wo.weight`, [D, H * cfg.v_head_dim], { ...g, tp: ROW() })
    return
  }

  const KV = cfg.attn_type === 'mha' ? H : cfg.n_kv_heads
  const hd = cfg.head_dim
  const kvTp = { style: 'kv', dim: 0, heads: KV }

  add(`${P}.wq.weight`, [H * hd, D], { ...g, tp: COL() })
  if (cfg.qkv_bias) add(`${P}.wq.bias`, [H * hd], { ...g, tp: COL() })
  add(`${P}.wk.weight`, [KV * hd, D], { ...g, tp: kvTp })
  if (cfg.qkv_bias) add(`${P}.wk.bias`, [KV * hd], { ...g, tp: kvTp })
  add(`${P}.wv.weight`, [KV * hd, D], { ...g, tp: kvTp })
  if (cfg.qkv_bias) add(`${P}.wv.bias`, [KV * hd], { ...g, tp: kvTp })
  add(`${P}.wo.weight`, [D, H * hd], { ...g, tp: ROW() })
  if (cfg.o_bias) add(`${P}.wo.bias`, [D], g)
  if (cfg.qk_norm) {
    add(`${P}.q_norm.weight`, [hd], { ...g, group: 'norm' })
    add(`${P}.k_norm.weight`, [hd], { ...g, group: 'norm' })
  }
  if (cfg.attn_sinks) add(`${P}.sinks`, [H], { ...g, tp: COL() })
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

  add(`${P}.router.weight`, [E, D], { ...lm, group: 'router' })
  if (cfg.router_bias) add(`${P}.router.bias`, [E], { ...lm, group: 'router' })
  if (cfg.balance_bias) {
    add(`${P}.router.balance_bias`, [E], { ...lm, group: 'router', kind: 'buffer', trainable: false })
  }

  const em = { ...lm, group: 'experts' }
  if (cfg.expert_layout === 'grouped') {
    const ex = { ...em, expert: { grouped: true } }
    add(`${P}.experts.w_gate`, [E, I, D], { ...ex, tp: COL(1) })
    if (cfg.expert_bias) add(`${P}.experts.w_gate_bias`, [E, I], { ...ex, tp: COL(1) })
    add(`${P}.experts.w_up`, [E, I, D], { ...ex, tp: COL(1) })
    if (cfg.expert_bias) add(`${P}.experts.w_up_bias`, [E, I], { ...ex, tp: COL(1) })
    add(`${P}.experts.w_down`, [E, D, I], { ...ex, tp: ROW(2) })
    if (cfg.expert_bias) add(`${P}.experts.w_down_bias`, [E, D], ex)
  } else {
    for (let e = 0; e < E; e++) {
      addSwiGLU(add, `${P}.experts.${e}`, D, I, cfg.expert_bias, { ...em, expert: { index: e, count: E } })
    }
  }

  if (cfg.n_shared_experts > 0) {
    addSwiGLU(add, `${P}.shared_experts`, D, I * cfg.n_shared_experts, cfg.expert_bias, {
      ...lm,
      group: 'shared_experts',
    })
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
  if (cfg.arch === 'moe') {
    if (cfg.n_activated_experts > cfg.n_routed_experts) {
      errors.push('n_activated_experts cannot exceed n_routed_experts')
    }
    if (cfg.n_dense_layers >= cfg.n_layers) {
      warnings.push('n_dense_layers ≥ n_layers: no MoE layers remain')
    }
  }
  return { errors, warnings }
}
