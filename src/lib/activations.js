// Forward activation shapes on one rank for a micro-batch. This is a shape
// walkthrough of the main tensors (what CP / SP / TP / EP do to the sequence,
// head and token axes), not a full activation-memory model.

import { DTYPE_BYTES, prod } from './format.js'
import { isMoeLayer } from './model.js'

const cdiv = (a, b) => Math.ceil(a / b)

export function seqDims(par, train) {
  const tpOn = par.tp > 1
  const sCp = cdiv(train.seq_len, par.cp)
  const sp = par.sp && tpOn
  return { sCp, sSp: sp ? cdiv(sCp, par.tp) : sCp, sp }
}

export function kvLocalHeads(cfg, par) {
  if (cfg.attn_type === 'mla') return cfg.n_heads / par.tp
  const KV = cfg.attn_type === 'mha' ? cfg.n_heads : cfg.n_kv_heads
  return KV % par.tp === 0 ? KV / par.tp : 1
}

// Whether the MoE block runs inside a TP region: tokens are gathered over TP
// (SP) before routing because routed or shared experts are TP-sharded.
export function moeTpRegion(cfg, par) {
  return par.tp > 1 && (par.tpExperts || cfg.n_shared_experts > 0)
}

export function moeTokens(cfg, par, train) {
  const { sCp, sSp } = seqDims(par, train)
  const T = train.micro_batch * (moeTpRegion(cfg, par) ? sCp : sSp)
  const k = cfg.n_activated_experts
  const E = cfg.n_routed_experts
  return { T, dispatched: T * k, perExpert: (T * k * par.ep) / E, localExperts: E / par.ep }
}

export function activationSections(cfg, par, train, prec) {
  const b = train.micro_batch
  const S = train.seq_len
  const D = cfg.dim
  const tp = par.tp
  const { sCp, sSp, sp } = seqDims(par, train)
  const bytes = DTYPE_BYTES[prec.compute]
  const row = (name, shape, note = '', elBytes = bytes) => ({ name, shape, note, bytes: prod(shape) * elBytes })
  const spNote = sp ? 'SP region: sequence sharded over TP' : ''
  const cpNote = par.cp > 1 ? `sequence sharded over CP (${S} / ${par.cp})` : ''
  const sections = []

  const emb = [
    row('tokens (input ids, int64)', [b, sCp], cpNote, 8),
    row('tok_embeddings → h', [b, sSp, D], [spNote, cpNote].filter(Boolean).join('; ')),
  ]
  sections.push({ title: 'Embedding', rows: emb })

  for (let i = 0; i < cfg.n_layers; i++) {
    const L = `layers.${i}`
    const rows = []
    rows.push(row(`${L}.attn_norm → x`, [b, sSp, D], spNote))
    if (sp) rows.push(row('all-gather(seq) → attn input', [b, sCp, D], 'SP → TP region'))

    const H = cfg.n_heads / tp
    const KVh = kvLocalHeads(cfg, par)
    let kShape
    let vShape
    let outDim
    if (cfg.attn_type === 'mla') {
      const qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
      if (cfg.q_lora_rank > 0) rows.push(row(`${L}.attn.q_a_proj → c_q`, [b, sCp, cfg.q_lora_rank], 'replicated over TP'))
      rows.push(row('q', [b, H, sCp, qk], tp > 1 ? `heads ${cfg.n_heads} / tp ${tp}` : ''))
      rows.push(row(`${L}.attn.kv_a_proj → c_kv ⊕ k_rope`, [b, sCp, cfg.kv_lora_rank + cfg.qk_rope_head_dim], 'compressed latent (what an MLA KV cache stores)'))
      kShape = [b, H, sCp, qk]
      vShape = [b, H, sCp, cfg.v_head_dim]
      outDim = H * cfg.v_head_dim
    } else {
      const hd = cfg.head_dim
      rows.push(row('q', [b, H, sCp, hd], tp > 1 ? `heads ${cfg.n_heads} / tp ${tp}` : ''))
      kShape = [b, KVh, sCp, hd]
      vShape = [b, KVh, sCp, hd]
      outDim = H * hd
    }
    rows.push(row('k', kShape))
    rows.push(row('v', vShape))
    if (par.cp > 1) {
      if (par.cpStyle === 'ring') {
        rows.push(row('ring recv: k chunk from peer', kShape, `× ${par.cp - 1} ring steps, one chunk live at a time`))
        rows.push(row('ring recv: v chunk from peer', vShape, `× ${par.cp - 1} ring steps`))
      } else {
        const full = (s) => [s[0], s[1], S, s[3]]
        rows.push(row('all-gathered k (full sequence)', full(kShape), 'all-gather KV over CP'))
        rows.push(row('all-gathered v (full sequence)', full(vShape), 'all-gather KV over CP'))
      }
    }
    rows.push(row('sdpa output', [b, sCp, outDim]))
    rows.push(row(`${L}.attn.wo → out`, sp ? [b, sSp, D] : [b, sCp, D], sp ? 'reduce-scatter(seq) back to SP region' : tp > 1 ? 'all-reduce over TP' : ''))

    const moe = isMoeLayer(cfg, i)
    const normName = moe ? 'moe_norm' : 'mlp_norm'
    rows.push(row(`${L}.${normName} → x`, [b, sSp, D], spNote))
    if (!moe) {
      if (sp) rows.push(row('all-gather(seq) → mlp input', [b, sCp, D], 'SP → TP region'))
      const I = cdiv(cfg.ffn_dim, tp)
      rows.push(row(`${L}.mlp.w_gate / w_up → h`, [b, sCp, I], tp > 1 ? 'column-parallel' : ''))
      rows.push(row(`${L}.mlp.w_down → out`, sp ? [b, sSp, D] : [b, sCp, D], sp ? 'reduce-scatter(seq)' : tp > 1 ? 'all-reduce over TP' : ''))
    } else {
      const { T, dispatched, perExpert, localExperts } = moeTokens(cfg, par, train)
      const E = cfg.n_routed_experts
      const Ie = par.tpExperts ? cdiv(cfg.moe_inter_dim, tp) : cfg.moe_inter_dim
      const region = moeTpRegion(cfg, par)
      if (sp && region) rows.push(row('all-gather(seq) → moe input', [b, sCp, D], 'SP → TP region'))
      rows.push(row('tokens flattened', [T, D], sp && !region ? 'tokens = b × seq/(cp·tp)' : ''))
      rows.push(row(`${L}.moe.router → scores`, [T, E]))
      rows.push(row('top-k indices (int64)', [T, cfg.n_activated_experts], '', 8))
      rows.push(row('top-k weights', [T, cfg.n_activated_experts]))
      if (par.ep > 1) {
        rows.push(row('all-to-all dispatch (send)', [dispatched, D], `EP ${par.ep}: T × k token copies`))
        rows.push(row('all-to-all dispatch (recv, balanced)', [dispatched, D], `${localExperts} local experts × ~${formatNum(perExpert)} tokens each`))
      } else {
        rows.push(row('permuted tokens', [dispatched, D], `${E} experts × ~${formatNum(perExpert)} tokens each`))
      }
      rows.push(row(`${L}.moe.experts w_gate / w_up → h`, [dispatched, Ie], par.tpExperts && tp > 1 ? 'expert TP column-parallel' : ''))
      rows.push(row(`${L}.moe.experts w_down → y`, [dispatched, D]))
      if (par.ep > 1) rows.push(row('all-to-all combine (recv)', [dispatched, D], 'back to source ranks'))
      if (cfg.n_shared_experts > 0) {
        rows.push(row(`${L}.moe.shared_experts → h`, [T, cdiv(cfg.moe_inter_dim * cfg.n_shared_experts, tp)]))
      }
      rows.push(row(`${L}.moe → out`, [b, sSp, D], sp && region ? 'reduce-scatter(seq) back to SP region' : region ? 'all-reduce over TP' : ''))
    }
    sections.push({ title: `${L} (${moe ? 'MoE' : 'dense'})`, rows })
  }

  const V = par.vocabParallel && tp > 1 ? cdiv(cfg.vocab_size, tp) : cfg.vocab_size
  sections.push({
    title: 'Head',
    rows: [
      row('final_norm → x', [b, sSp, D], spNote),
      row('lm_head → logits', [b, sCp, V], par.vocabParallel && tp > 1 ? 'vocab-parallel logits (loss parallel)' : ''),
    ],
  })
  return sections
}

function formatNum(x) {
  return Number.isInteger(x) ? x.toLocaleString('en-US') : x.toFixed(1)
}
