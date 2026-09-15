// Forward activation shapes on one rank for a micro-batch. This is a shape
// walkthrough of the main tensors (what CP / SP / TP / EP do to the sequence,
// head and token axes), not a full activation-memory model.

import { DTYPE_BYTES, prod } from './format.js'
import { compressRatio, hasIndexer, isHashLayer, isMoeLayer, layerMixer } from './model.js'

const cdiv = (a, b) => Math.ceil(a / b)

export function seqDims(par, train) {
  const tpOn = par.tp > 1
  const sCp = cdiv(train.seq_len, par.cp)
  const sp = par.sp && tpOn
  return { sCp, sSp: sp ? cdiv(sCp, par.tp) : sCp, sp }
}

export function kvLocalHeads(cfg, par) {
  if (cfg.attn_type === 'dsv4') return 1
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
  // routed tokens travel at the latent width when the experts use one (Kimi K3)
  const width = cfg.moe_latent_dim > 0 ? cfg.moe_latent_dim : cfg.dim
  return { T, dispatched: T * k, perExpert: (T * k * par.ep) / E, localExperts: E / par.ep, width }
}

// Recurrent state of a linear-attention layer on one TP rank (KDA / gated DeltaNet).
export function linearState(cfg, par) {
  const kda = cfg.linear_attn === 'kda'
  const Hv = cdiv(cfg.la_num_heads, par.tp)
  const dk = cfg.la_head_dim
  const dv = kda ? cfg.la_head_dim : cfg.la_v_head_dim
  const convCh = kda ? 3 * cfg.la_num_heads * dk : 2 * cfg.la_num_k_heads * dk + cfg.la_num_heads * dv
  return { Hv, Hk: kda ? Hv : cdiv(cfg.la_num_k_heads, par.tp), dk, dv, state: Hv * dk * dv, conv: cdiv(convCh, par.tp) * (cfg.la_conv - 1) }
}

// What context parallelism exchanges for layer i's token mixer in one
// micro-batch. Entries carry a short `what` (comms table) and a `note`.
export function cpExchange(cfg, par, train, i, bytesPer) {
  if (par.cp <= 1) return null
  const b = train.micro_batch
  const S = train.seq_len
  const { sCp } = seqDims(par, train)
  const steps = par.cp - 1
  const mixer = layerMixer(cfg, i)

  if (mixer === 'kda' || mixer === 'gdn') {
    const st = linearState(cfg, par)
    const bytes = b * (st.state + st.conv) * bytesPer
    return {
      title: 'state hand-off',
      sub: 'context parallel · linear attention',
      fwd: [{ op: 'send/recv', count: 1, bytes, what: 'linear attention: recurrent + conv state', note: 'chunks run in sequence order: receive the recurrent state [b, h, dk, dv] and short-conv tail from the previous CP rank, pass the final state to the next' }],
      bwd: [{ op: 'send/recv', count: 1, bytes, what: 'linear attention: state grad', note: 'the state gradient flows back to the previous CP rank' }],
    }
  }

  if (mixer === 'dsv4') {
    const hd = cfg.head_dim
    const r = compressRatio(cfg, i)
    const halo = b * Math.min(cfg.window_size, sCp) * hd * bytesPer
    const fwd = [{ op: 'send/recv', count: 1, bytes: halo, what: 'sliding-window KV halo', note: `the last ${cfg.window_size} window KV entries go to the next CP rank so its first tokens still see a full window` }]
    const bwd = [{ op: 'send/recv', count: 1, bytes: halo, what: 'window KV halo grad', note: 'gradients of the borrowed window KV return to their owner' }]
    if (r) {
      let numel = b * cdiv(S, r) * hd
      if (r === 4 && cfg.index_n_heads > 0) numel += b * cdiv(S, 4) * cfg.index_head_dim
      const bytes = numel * bytesPer
      fwd.push({ op: 'all-gather', count: 1, bytes, what: `compressed KV ×${r}`, note: `gather the ×${r} compressed KV${r === 4 ? ' and indexer keys' : ''} of the whole sequence; each rank compresses its own chunk` })
      bwd.push({ op: 'reduce-scatter', count: 1, bytes, what: `compressed KV ×${r} grad`, note: 'sum compressed-KV grads and return each chunk’s share to its owner' })
    }
    return { title: 'sparse attention KV', sub: 'context parallel', fwd, bwd }
  }

  let kvNumel
  if (mixer === 'mla') {
    kvNumel = b * cdiv(cfg.n_heads, par.tp) * sCp * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
    if (hasIndexer(cfg, i)) kvNumel += b * sCp * cfg.index_head_dim // indexer keys travel with K,V
  } else {
    kvNumel = 2 * b * kvLocalHeads(cfg, par) * sCp * cfg.head_dim
  }
  const kv = kvNumel * bytesPer
  if (par.cpStyle === 'ring') {
    return {
      title: 'ring attention',
      sub: 'context parallel',
      fwd: [{ op: 'send/recv', count: steps, bytes: kv, what: 'ring attention: K,V chunk per step', note: `pass this rank’s K,V chunk to the next CP rank and receive one from the previous; ${steps} step(s), attention is computed blockwise against each chunk as it arrives` }],
      bwd: [{ op: 'send/recv', count: steps, bytes: 2 * kv, what: 'ring attention: K,V + dK,dV per step', note: 'rotate K,V together with the accumulated dK,dV around the ring' }],
    }
  }
  return {
    title: 'all-gather KV attention',
    sub: 'context parallel',
    fwd: [{ op: 'all-gather', count: 1, bytes: kv * par.cp, what: 'all-gather K,V (full sequence)', note: `gather K,V${hasIndexer(cfg, i) ? ' and indexer keys' : ''} for the full sequence; the local q chunk attends to every key (Llama 3 style)` }],
    bwd: [
      { op: 'all-gather', count: 1, bytes: kv * par.cp, what: 're-gather K,V', note: 're-gather K,V for backward' },
      { op: 'reduce-scatter', count: 1, bytes: kv * par.cp, what: 'dK,dV back to owners', note: 'sum dK,dV and return them to the ranks owning each chunk' },
    ],
  }
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
    const mixer = layerMixer(cfg, i)
    if (cfg.attn_type === 'dsv4') rows.push(row(`${L} residual stream (hyper-connections)`, [b, sSp, cfg.hc_mult, D], `${cfg.hc_mult} residual copies mixed by hc_pre / hc_post`))
    if (cfg.attn_res) rows.push(row(`${L}.attn residual mix → x`, [b, sSp, D], 'softmax-weighted mix of earlier layer outputs'))
    rows.push(row(`${L}.attn_norm → x`, [b, sSp, D], spNote))
    if (sp) rows.push(row('all-gather(seq) → attn input', [b, sCp, D], 'SP → TP region'))

    const H = cdiv(cfg.n_heads, tp)
    const KVh = kvLocalHeads(cfg, par)
    let kShape
    let vShape
    let outDim
    let fullAttn = true
    if (mixer === 'kda' || mixer === 'gdn') {
      fullAttn = false
      const st = linearState(cfg, par)
      if (mixer === 'kda') {
        for (const x of ['q', 'k', 'v']) rows.push(row(`${L}.attn.${x}_proj + conv → ${x}`, [b, sCp, st.Hv * st.dk], tp > 1 ? 'column-parallel heads' : ''))
        rows.push(row('decay α (low-rank)', [b, sCp, st.Hv * st.dk]))
        rows.push(row('output gate g', [b, sCp, st.Hv * st.dv]))
      } else {
        rows.push(row(`${L}.attn.in_proj_qkv + conv → q,k,v`, [b, sCp, cdiv(2 * cfg.la_num_k_heads * st.dk + cfg.la_num_heads * st.dv, tp)], tp > 1 ? 'column-parallel heads' : ''))
        rows.push(row(`${L}.attn.in_proj_z → z gate`, [b, sCp, st.Hv * st.dv]))
        rows.push(row('decay α', [b, sCp, st.Hv]))
      }
      rows.push(row('write strength β', [b, sCp, st.Hv]))
      rows.push(row('recurrent state S (per sequence)', [b, st.Hv, st.dk, st.dv], 'size does not grow with sequence length'))
      if (par.cp > 1) rows.push(row('state from previous CP rank', [b, st.Hv, st.dk, st.dv], 'chunks are processed in sequence order'))
      rows.push(row('gated norm → o', [b, sCp, st.Hv * st.dv]))
      outDim = st.Hv * st.dv
    } else if (mixer === 'dsv4') {
      fullAttn = false
      const hd = cfg.head_dim
      const r = compressRatio(cfg, i)
      rows.push(row(`${L}.attn.wq_a → qr`, [b, sCp, cfg.q_lora_rank], 'replicated over TP'))
      rows.push(row('q', [b, sCp, H, hd], tp > 1 ? `heads ${cfg.n_heads} / tp ${tp}` : ''))
      rows.push(row(`${L}.attn.wkv → window kv`, [b, sCp, hd], 'one KV head, replicated over TP'))
      if (r) rows.push(row(`${L}.attn.compressor → kv ×${r}`, [b, cdiv(sCp, r), hd], 'gated pooling over consecutive tokens'))
      if (r && par.cp > 1) rows.push(row(`all-gathered compressed kv ×${r}`, [b, cdiv(S, r), hd], 'whole sequence, over CP'))
      if (r === 4 && cfg.index_n_heads > 0) {
        rows.push(row(`${L}.attn.indexer → q`, [b, sCp, cdiv(cfg.index_n_heads, tp), cfg.index_head_dim]))
        rows.push(row('index scores (fp32)', [b, sCp, cdiv(S, 4)], tp > 1 ? 'all-reduced over TP' : '', 4))
        rows.push(row('top-k block indices (int32)', [b, sCp, Math.min(cfg.index_topk, cdiv(S, 4))], 'sparse attention reads only these compressed blocks', 4))
      }
      rows.push(row('sparse attention output', [b, sCp, H * hd]))
      rows.push(row(`${L}.attn.wo_a → grouped low-rank`, [b, sCp, cdiv(cfg.o_groups, tp) * cfg.o_lora_rank], tp > 1 ? 'column-parallel groups' : ''))
    } else if (cfg.attn_type === 'mla') {
      const qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
      if (cfg.q_lora_rank > 0) rows.push(row(`${L}.attn.q_a_proj → c_q`, [b, sCp, cfg.q_lora_rank], 'replicated over TP'))
      rows.push(row('q', [b, H, sCp, qk], tp > 1 ? `heads ${cfg.n_heads} / tp ${tp}` : ''))
      rows.push(row(`${L}.attn.kv_a_proj → c_kv ⊕ k_rope`, [b, sCp, cfg.kv_lora_rank + cfg.qk_rope_head_dim], 'compressed latent (what an MLA KV cache stores)'))
      if (hasIndexer(cfg, i)) {
        rows.push(row(`${L}.attn.indexer → keys`, [b, sCp, cfg.index_head_dim], 'lightning indexer keys (replicated over TP)'))
        rows.push(row('indexer scores', [b, sCp, par.cp > 1 ? S : sCp], par.cp > 1 ? 'each local query scores every key in the sequence' : 'each query scores every earlier key'))
        rows.push(row('top-k key indices (int64)', [b, sCp, Math.min(cfg.index_topk || S, S)], 'sparse attention only reads these keys', 8))
      }
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
    if (fullAttn) rows.push(row('k', kShape))
    if (fullAttn) rows.push(row('v', vShape))
    if (fullAttn && cfg.attn_output_gate) rows.push(row('output gate σ(g)', [b, sCp, outDim], 'multiplies the attention output'))
    if (fullAttn && par.cp > 1) {
      if (par.cpStyle === 'ring') {
        rows.push(row('ring recv: k chunk from peer', kShape, `× ${par.cp - 1} ring steps, one chunk live at a time`))
        rows.push(row('ring recv: v chunk from peer', vShape, `× ${par.cp - 1} ring steps`))
      } else {
        const full = (s) => [s[0], s[1], S, s[3]]
        rows.push(row('all-gathered k (full sequence)', full(kShape), 'all-gather KV over CP'))
        rows.push(row('all-gathered v (full sequence)', full(vShape), 'all-gather KV over CP'))
      }
    }
    if (fullAttn) rows.push(row('sdpa output', [b, sCp, outDim]))
    rows.push(row(`${L}.attn.wo → out`, sp ? [b, sSp, D] : [b, sCp, D], sp ? 'reduce-scatter(seq) back to SP region' : tp > 1 ? 'all-reduce over TP' : ''))

    const moe = isMoeLayer(cfg, i)
    const normName = moe ? 'moe_norm' : 'mlp_norm'
    if (cfg.attn_res) rows.push(row(`${L}.mlp residual mix → x`, [b, sSp, D], 'softmax-weighted mix of earlier layer outputs'))
    rows.push(row(`${L}.${normName} → x`, [b, sSp, D], spNote))
    if (!moe) {
      if (sp) rows.push(row('all-gather(seq) → mlp input', [b, sCp, D], 'SP → TP region'))
      const I = cdiv(cfg.ffn_dim, tp)
      rows.push(row(`${L}.mlp.w_gate / w_up → h`, [b, sCp, I], tp > 1 ? 'column-parallel' : ''))
      rows.push(row(`${L}.mlp.w_down → out`, sp ? [b, sSp, D] : [b, sCp, D], sp ? 'reduce-scatter(seq)' : tp > 1 ? 'all-reduce over TP' : ''))
    } else {
      const { T, dispatched, perExpert, localExperts, width: Dx } = moeTokens(cfg, par, train)
      const E = cfg.n_routed_experts
      const Ie = par.tpExperts ? cdiv(cfg.moe_inter_dim, tp) : cfg.moe_inter_dim
      const region = moeTpRegion(cfg, par)
      if (sp && region) rows.push(row('all-gather(seq) → moe input', [b, sCp, D], 'SP → TP region'))
      rows.push(row('tokens flattened', [T, D], sp && !region ? 'tokens = b × seq/(cp·tp)' : ''))
      rows.push(row(`${L}.moe.router → scores`, [T, E]))
      rows.push(row(isHashLayer(cfg, i) ? 'hash-routed expert ids (tid2eid[token])' : 'top-k indices (int64)', [T, cfg.n_activated_experts], '', 8))
      rows.push(row('top-k weights', [T, cfg.n_activated_experts]))
      if (cfg.moe_latent_dim > 0) rows.push(row(`${L}.moe.latent_down → tokens`, [T, Dx], 'routed experts work at the latent width'))
      if (par.ep > 1) {
        rows.push(row('all-to-all dispatch (send)', [dispatched, Dx], `EP ${par.ep}: T × k token copies`))
        rows.push(row('all-to-all dispatch (recv, balanced)', [dispatched, Dx], `${localExperts} local experts × ~${formatNum(perExpert)} tokens each`))
      } else {
        rows.push(row('permuted tokens', [dispatched, Dx], `${E} experts × ~${formatNum(perExpert)} tokens each`))
      }
      rows.push(row(`${L}.moe.experts w_gate / w_up → h`, [dispatched, Ie], par.tpExperts && tp > 1 ? 'expert TP column-parallel' : ''))
      rows.push(row(`${L}.moe.experts w_down → y`, [dispatched, Dx]))
      if (par.ep > 1) rows.push(row('all-to-all combine (recv)', [dispatched, Dx], 'back to source ranks'))
      if (cfg.moe_latent_dim > 0) rows.push(row(`${L}.moe.latent_up → routed out`, [T, D]))
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
