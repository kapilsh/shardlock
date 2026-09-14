// One transformer block as a data-flow graph on a single rank: modules with
// their local params, collectives inline where they fire, and edges labeled
// with the local tensor shapes. Comm nodes carry both forward and backward
// entries so the same layout renders either pass (backward = arrows reversed,
// conjugate collectives, gradient shapes).
//
// Nodes sit on a 3-lane grid (col 0 / 1 / 2, fractional allowed); `row` is the
// vertical order. Layout and drawing live in LayerView.jsx.

import { DTYPE_BYTES, prod } from './format.js'
import { isMoeLayer } from './model.js'
import { kvLocalHeads, moeTokens, moeTpRegion, seqDims } from './activations.js'
import { placeParam, unshardedLocalNumel } from './shard.js'

const cdiv = (a, b) => Math.ceil(a / b)
const dm = (n, sym) => ({ n, sym })
const lbl = (shape, name) => ({ name, shape })

export function buildLayerFlow(cfg, par, train, prec, params, li, coords) {
  const b = train.micro_batch
  const S = train.seq_len
  const D = cfg.dim
  const tp = par.tp
  const tpOn = tp > 1
  const { sCp, sSp, sp } = seqDims(par, train)
  const cB = DTYPE_BYTES[prec.compute]
  const gB = DTYPE_BYTES[prec.grad]
  const L = `layers.${li}`
  const moe = isMoeLayer(cfg, li)
  const byFqn = new Map(params.map((p) => [p.fqn, p]))
  const pick = (...fqns) => fqns.filter((f) => byFqn.has(f))

  const nodes = []
  const edges = []
  let row = -1
  const nextRow = () => ++row
  const node = (n) => {
    nodes.push({ col: 1, row, fwd: [], bwd: [], ...n })
    return n.id
  }
  const edge = (from, to, labels = [], opts = {}) => edges.push({ from, to, labels, ...opts })

  const cpSym = par.cp > 1 ? 's/cp' : 's'
  const sym = {
    b: dm(b, 'b'),
    S: dm(S, 's'),
    sCp: dm(sCp, cpSym),
    sSp: dm(sSp, sp ? (par.cp > 1 ? 's/(cp·tp)' : 's/tp') : cpSym),
    d: dm(D, 'd'),
  }
  const hidden = (seq) => [sym.b, seq, sym.d]
  const bytesOf = (shape) => prod(shape.map((x) => x.n)) * cB
  const perTp = (n, name) => dm(cdiv(n, tp), tpOn ? `${name}/tp` : name)

  // ---- TP region boundaries (Megatron f / g, or SP all-gather / reduce-scatter)
  const tpGroup = `TP·${tp}`
  const tpEntry = (op, bytes, note, idle = false) => ({ axis: 'TP', op, group: tpGroup, groupKey: 'tp', bytes: idle ? 0 : bytes, note, idle })
  const tpEnter = (id, where) => {
    const bytes = bytesOf(hidden(sym.sCp))
    node({
      id,
      kind: 'comm',
      title: `enter TP region · ${where}`,
      sub: sp ? 'sequence parallel → tensor parallel' : 'Megatron f',
      fwd: [
        sp
          ? tpEntry('all-gather', bytes, `SP: gather sequence shards (${sym.sSp.sym} → ${sym.sCp.sym}) so the column-parallel layers see the whole local sequence`)
          : tpEntry('identity', bytes, 'f: the input is already identical on every TP rank', true),
      ],
      bwd: [
        sp
          ? tpEntry('reduce-scatter', bytes, 'sum input grads over TP ranks and scatter them back to sequence shards')
          : tpEntry('all-reduce', bytes, 'f backward: sum the input grads produced by each column-parallel shard'),
      ],
    })
    return id
  }
  const tpExit = (id, where) => {
    const bytes = bytesOf(hidden(sym.sCp))
    node({
      id,
      kind: 'comm',
      title: `exit TP region · ${where}`,
      sub: sp ? 'tensor parallel → sequence parallel' : 'Megatron g',
      fwd: [
        sp
          ? tpEntry('reduce-scatter', bytes, 'sum row-parallel outputs over TP and keep only this rank’s sequence shard')
          : tpEntry('all-reduce', bytes, 'g: sum the row-parallel partial outputs'),
      ],
      bwd: [
        sp
          ? tpEntry('all-gather', bytes, 'gather output grads for the full local sequence')
          : tpEntry('identity', bytes, 'g backward: output grad is already identical on every TP rank', true),
      ],
    })
    return id
  }

  // ---- Data-parallel param / grad collectives for this layer's FSDP unit ----
  const layerParams = params.filter((p) => p.layer === li)
  const trainable = layerParams.filter((p) => p.kind === 'param')
  const sum = (ps, f) => ps.reduce((a, p) => a + f(p), 0)
  const full = (ps) => sum(ps, (p) => unshardedLocalNumel(p, par, coords))
  const local = (ps) => sum(ps, (p) => placeParam(p, par, coords).localNumel)
  const denseP = trainable.filter((p) => !p.expert)
  const expertP = trainable.filter((p) => p.expert)
  const top = { fwd: [], bwd: [] }
  const bottom = { fwd: [], bwd: [] }
  const edpDeg = par.dp / par.ep

  if (par.dpStrategy === 'ddp') {
    if (par.dp > 1 && denseP.length) {
      top.bwd.push({ axis: 'DP', op: 'all-reduce', group: `DP·${par.dp}`, groupKey: 'dp', bytes: full(denseP) * gB, note: 'average grads across replicas as soon as they are ready (bucketed with neighbouring layers)' })
    }
    if (expertP.length && edpDeg > 1) {
      top.bwd.push({ axis: 'DP', op: 'all-reduce', group: `EDP·${edpDeg}`, groupKey: 'edp', bytes: full(expertP) * gB, note: 'average expert grads across expert-data-parallel replicas' })
    }
  } else {
    const hsdp = par.dpStrategy === 'hsdp'
    const axis = hsdp ? 'HSDP' : 'FSDP'
    const parts = [
      { ps: denseP, deg: hsdp ? par.hsdpShard : par.dp, what: 'params', key: 'shard' },
      { ps: expertP, deg: hsdp ? par.hsdpShard / par.ep : edpDeg, what: 'expert params', key: hsdp ? null : 'edp' },
    ]
    for (const part of parts) {
      if (!part.ps.length || part.deg <= 1) continue
      const group = `${axis}·${part.deg}`
      const e = { axis, group, groupKey: part.key }
      top.fwd.push({ ...e, op: 'all-gather', bytes: full(part.ps) * cB, note: `unshard ${L} ${part.what} (${prec.compute}) right before this layer’s forward` })
      top.bwd.push({ ...e, op: 'reduce-scatter', bytes: full(part.ps) * gB, note: `average ${part.what} grads across the shard group and keep only this rank’s shard` })
      if (par.reshardAfterForward) {
        bottom.fwd.push({ ...e, op: 'reshard', idle: true, bytes: 0, note: 'free the gathered params once forward leaves the layer' })
        bottom.bwd.push({ ...e, op: 'all-gather', bytes: full(part.ps) * cB, note: `re-unshard ${part.what} before this layer’s backward` })
      }
    }
    if (hsdp && par.dp / par.hsdpShard > 1) {
      const rep = par.dp / par.hsdpShard
      top.bwd.push({ axis, op: 'all-reduce', group: `replica·${rep}`, groupKey: null, bytes: local(trainable) * gB, note: 'average the sharded grads across HSDP replica blocks' })
    }
  }

  // ---- Block input ------------------------------------------------------------
  nextRow()
  node({ id: 'in', kind: 'io', title: `${L} input`, sub: li === 0 ? 'from tok_embeddings' : `from layers.${li - 1}` })
  if (top.fwd.length || top.bwd.length) {
    node({ id: 'dp_top', kind: 'comm', col: 2, title: `${L} params · ${par.dpStrategy.toUpperCase()}`, sub: 'parameter / gradient collectives', ...top })
  }

  // ---- Attention --------------------------------------------------------------
  nextRow()
  node({ id: 'res1', kind: 'fork' })
  edge('in', 'res1', [lbl(hidden(sym.sSp), 'x')])
  nextRow()
  node({ id: 'attn_norm', kind: 'module', title: 'attn_norm', sub: 'RMSNorm', params: pick(`${L}.attn_norm.weight`) })
  edge('res1', 'attn_norm')
  let attnIn = 'attn_norm'
  if (tpOn) {
    nextRow()
    tpEnter('tp_attn_in', 'attention')
    edge('attn_norm', 'tp_attn_in', [lbl(hidden(sym.sSp))])
    attnIn = 'tp_attn_in'
  }
  const inLbl = [lbl(hidden(sym.sCp))]
  const H = perTp(cfg.n_heads, 'h')
  const A = `${L}.attn`
  let qOut
  let qLabels
  let kvSources
  let kvElems

  if (cfg.attn_type !== 'mla') {
    const hd = dm(cfg.head_dim, 'hd')
    const KVn = cfg.attn_type === 'mha' ? cfg.n_heads : cfg.n_kv_heads
    const KVsym = cfg.attn_type === 'mha' ? 'h' : 'kv'
    const kvRep = tpOn && KVn % tp !== 0
    const kvh = dm(kvLocalHeads(cfg, par), tpOn ? (kvRep ? '1' : `${KVsym}/tp`) : KVsym)
    const qShape = [sym.b, sym.sCp, H, hd]
    const kShape = [sym.b, sym.sCp, kvh, hd]
    const colSub = tpOn ? 'column-parallel' : 'linear'
    nextRow()
    node({ id: 'wq', col: 0, kind: 'module', title: 'wq', sub: colSub, params: pick(`${A}.wq.weight`, `${A}.wq.bias`) })
    node({ id: 'wk', col: 1, kind: 'module', title: 'wk', sub: kvRep ? 'KV heads replicated over TP' : colSub, params: pick(`${A}.wk.weight`, `${A}.wk.bias`) })
    node({ id: 'wv', col: 2, kind: 'module', title: 'wv', sub: kvRep ? 'KV heads replicated over TP' : colSub, params: pick(`${A}.wv.weight`, `${A}.wv.bias`) })
    edge(attnIn, 'wq')
    edge(attnIn, 'wk', inLbl)
    edge(attnIn, 'wv')
    let qPrev = 'wq'
    let kPrev = 'wk'
    if (cfg.qk_norm) {
      nextRow()
      node({ id: 'q_norm', col: 0, kind: 'module', title: 'q_norm', sub: 'RMSNorm per head', params: pick(`${A}.q_norm.weight`) })
      node({ id: 'k_norm', col: 1, kind: 'module', title: 'k_norm', sub: 'RMSNorm per head', params: pick(`${A}.k_norm.weight`) })
      edge('wq', 'q_norm', [lbl(qShape, 'q')])
      edge('wk', 'k_norm', [lbl(kShape, 'k')])
      qPrev = 'q_norm'
      kPrev = 'k_norm'
    }
    nextRow()
    node({ id: 'rope_q', col: 0, kind: 'op', title: 'RoPE', sub: 'rotate q' })
    node({ id: 'rope_k', col: 1, kind: 'op', title: 'RoPE', sub: 'rotate k' })
    edge(qPrev, 'rope_q', [lbl(qShape, 'q')])
    edge(kPrev, 'rope_k', [lbl(kShape, 'k')])
    qOut = 'rope_q'
    qLabels = [lbl(qShape, 'q')]
    kvSources = [
      ['rope_k', [lbl(kShape, 'k')]],
      ['wv', [lbl(kShape, 'v')]],
    ]
    kvElems = 2 * prod(kShape.map((x) => x.n))
  } else {
    const qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    const qkd = dm(qk, 'qk_hd')
    const vd = dm(cfg.v_head_dim, 'v_hd')
    const ropeD = dm(cfg.qk_rope_head_dim, 'rope_hd')
    const kvr = dm(cfg.kv_lora_rank, 'kv_rank')
    const one = dm(1, '1')
    const lora = cfg.q_lora_rank > 0
    const qr = dm(cfg.q_lora_rank, 'q_rank')
    const qFlat = [sym.b, sym.sCp, dm(H.n * qk, `${H.sym}·qk_hd`)]

    nextRow() // A
    if (lora) node({ id: 'q_a', col: 0, kind: 'module', title: 'q_a_proj', sub: 'low-rank down · replicated', params: pick(`${A}.q_a_proj.weight`) })
    else node({ id: 'q_a', col: 0, kind: 'module', title: 'wq', sub: tpOn ? 'column-parallel' : 'linear', params: pick(`${A}.wq.weight`) })
    node({ id: 'kv_a', col: 1, kind: 'module', title: 'kv_a_proj', sub: 'low-rank down · replicated', params: pick(`${A}.kv_a_proj.weight`) })
    edge(attnIn, 'q_a')
    edge(attnIn, 'kv_a', inLbl)
    nextRow() // B
    if (lora) node({ id: 'q_a_norm', col: 0, kind: 'module', title: 'q_a_norm', sub: 'RMSNorm', params: pick(`${A}.q_a_norm.weight`) })
    node({ id: 'kv_split', col: 1, kind: 'op', title: 'split', sub: 'c_kv | k_rope' })
    if (lora) edge('q_a', 'q_a_norm', [lbl([sym.b, sym.sCp, qr])])
    edge('kv_a', 'kv_split', [lbl([sym.b, sym.sCp, dm(cfg.kv_lora_rank + cfg.qk_rope_head_dim, 'kv_rank+rope_hd')])])
    nextRow() // C
    if (lora) node({ id: 'q_b', col: 0, kind: 'module', title: 'q_b_proj', sub: tpOn ? 'column-parallel' : 'up proj', params: pick(`${A}.q_b_proj.weight`) })
    node({ id: 'kv_a_norm', col: 1, kind: 'module', title: 'kv_a_norm', sub: 'RMSNorm', params: pick(`${A}.kv_a_norm.weight`) })
    node({ id: 'rope_k', col: 2, kind: 'op', title: 'RoPE', sub: 'shared k_rope (1 head)' })
    if (lora) edge('q_a_norm', 'q_b', [lbl([sym.b, sym.sCp, qr])])
    edge('kv_split', 'kv_a_norm', [lbl([sym.b, sym.sCp, kvr], 'c_kv')])
    edge('kv_split', 'rope_k', [lbl([sym.b, sym.sCp, one, ropeD], 'k_rope')], { labelAt: 'mid' })
    nextRow() // D
    node({ id: 'rope_q', col: 0, kind: 'op', title: 'split + RoPE', sub: 'q_nope | RoPE(q_rope)' })
    node({ id: 'kv_b', col: 1, kind: 'module', title: 'kv_b_proj', sub: tpOn ? 'column-parallel' : 'up proj', params: pick(`${A}.kv_b_proj.weight`) })
    edge(lora ? 'q_b' : 'q_a', 'rope_q', [lbl(qFlat)])
    edge('kv_a_norm', 'kv_b', [lbl([sym.b, sym.sCp, kvr])])
    nextRow() // E
    node({ id: 'kv_join', col: 1, kind: 'op', title: 'split + concat', sub: 'k = [k_nope, k_rope], v' })
    edge('kv_b', 'kv_join', [lbl([sym.b, sym.sCp, dm(H.n * (cfg.qk_nope_head_dim + cfg.v_head_dim), `${H.sym}·(nope+v_hd)`)])])
    edge('rope_k', 'kv_join')
    qOut = 'rope_q'
    qLabels = [lbl([sym.b, sym.sCp, H, qkd], 'q')]
    kvSources = [['kv_join', [lbl([sym.b, sym.sCp, H, qkd], 'k'), lbl([sym.b, sym.sCp, H, vd], 'v')]]]
    kvElems = b * H.n * sCp * (qk + cfg.v_head_dim)
  }

  let sdpaIn = kvSources
  const ring = par.cpStyle === 'ring'
  if (par.cp > 1) {
    nextRow()
    const group = `CP·${par.cp}`
    const kvBytes = kvElems * cB
    const steps = par.cp - 1
    const e = { axis: 'CP', group, groupKey: 'cp' }
    node({
      id: 'cp',
      col: kvSources.length === 2 ? 1.5 : 1,
      kind: 'comm',
      title: ring ? 'ring attention' : 'all-gather KV attention',
      sub: 'context parallel',
      fwd: ring
        ? [{ ...e, op: 'send/recv', count: steps, bytes: kvBytes, note: `pass this rank’s K,V chunk to the next CP rank and receive one from the previous; ${steps} step(s), attention is computed blockwise against each chunk as it arrives` }]
        : [{ ...e, op: 'all-gather', bytes: kvBytes * par.cp, note: 'gather K,V for the full sequence; the local q chunk attends to every key (Llama 3 style)' }],
      bwd: ring
        ? [{ ...e, op: 'send/recv', count: steps, bytes: 2 * kvBytes, note: 'rotate K,V together with the accumulated dK,dV around the ring' }]
        : [
            { ...e, op: 'all-gather', bytes: kvBytes * par.cp, note: 're-gather K,V for backward' },
            { ...e, op: 'reduce-scatter', bytes: kvBytes * par.cp, note: 'sum dK,dV and return them to the ranks owning each chunk' },
          ],
    })
    for (const [src, labels] of kvSources) edge(src, 'cp', labels)
    const outLabels = kvSources
      .flatMap(([, labels]) => labels)
      .map((l) => (ring ? { ...l, name: `${l.name} chunk` } : { ...l, shape: l.shape.map((x, i) => (i === 1 ? sym.S : x)) }))
    sdpaIn = [['cp', outLabels]]
  }

  nextRow()
  node({
    id: 'sdpa',
    kind: 'op',
    title: 'scaled dot-product attention',
    sub: par.cp > 1 ? (ring ? 'causal · blockwise over the ring' : 'causal · local q vs all k,v') : 'causal · flash',
    params: pick(`${A}.sinks`),
  })
  edge(qOut, 'sdpa', qLabels)
  for (const [src, labels] of sdpaIn) edge(src, 'sdpa', labels)
  const outDim = cfg.attn_type === 'mla' ? dm(H.n * cfg.v_head_dim, `${H.sym}·v_hd`) : dm(H.n * cfg.head_dim, `${H.sym}·hd`)
  nextRow()
  node({ id: 'wo', kind: 'module', title: 'wo', sub: tpOn ? 'row-parallel' : 'output proj', params: pick(`${A}.wo.weight`, `${A}.wo.bias`) })
  edge('sdpa', 'wo', [lbl([sym.b, sym.sCp, outDim])])
  let attnOut = 'wo'
  if (tpOn) {
    nextRow()
    tpExit('tp_attn_out', 'attention')
    edge('wo', 'tp_attn_out', [lbl(hidden(sym.sCp))])
    attnOut = 'tp_attn_out'
  }
  nextRow()
  node({ id: 'add1', kind: 'add' })
  edge(attnOut, 'add1', [lbl(hidden(sym.sSp))])
  edges.push({ from: 'res1', to: 'add1', kind: 'residual', labels: [] })

  // ---- FFN ----------------------------------------------------------------------
  nextRow()
  node({ id: 'res2', kind: 'fork' })
  edge('add1', 'res2')
  let ffnOut
  let ffnOutShape
  let exitNeeded

  if (!moe) {
    const F = `${L}.mlp`
    nextRow()
    node({ id: 'ffn_norm', kind: 'module', title: 'mlp_norm', sub: 'RMSNorm', params: pick(`${L}.mlp_norm.weight`) })
    edge('res2', 'ffn_norm')
    let mIn = 'ffn_norm'
    if (tpOn) {
      nextRow()
      tpEnter('tp_mlp_in', 'mlp')
      edge('ffn_norm', 'tp_mlp_in', [lbl(hidden(sym.sSp))])
      mIn = 'tp_mlp_in'
    }
    const I = perTp(cfg.ffn_dim, 'ffn')
    const hI = [sym.b, sym.sCp, I]
    nextRow()
    node({ id: 'w_gate', col: 0, kind: 'module', title: 'w_gate', sub: tpOn ? 'column-parallel' : 'gate proj', params: pick(`${F}.w_gate.weight`, `${F}.w_gate.bias`) })
    node({ id: 'w_up', col: 2, kind: 'module', title: 'w_up', sub: tpOn ? 'column-parallel' : 'up proj', params: pick(`${F}.w_up.weight`, `${F}.w_up.bias`) })
    edge(mIn, 'w_gate', [lbl(hidden(sym.sCp))])
    edge(mIn, 'w_up')
    nextRow()
    node({ id: 'act', kind: 'op', title: 'SiLU(gate) ⊙ up', sub: 'SwiGLU' })
    edge('w_gate', 'act', [lbl(hI)])
    edge('w_up', 'act')
    nextRow()
    node({ id: 'w_down', kind: 'module', title: 'w_down', sub: tpOn ? 'row-parallel' : 'down proj', params: pick(`${F}.w_down.weight`, `${F}.w_down.bias`) })
    edge('act', 'w_down', [lbl(hI)])
    ffnOut = 'w_down'
    ffnOutShape = hidden(sym.sCp)
    exitNeeded = tpOn
  } else {
    const M = `${L}.moe`
    nextRow()
    node({ id: 'ffn_norm', kind: 'module', title: 'moe_norm', sub: 'RMSNorm', params: pick(`${L}.moe_norm.weight`) })
    edge('res2', 'ffn_norm')
    const region = moeTpRegion(cfg, par)
    let mIn = 'ffn_norm'
    if (region) {
      nextRow()
      tpEnter('tp_moe_in', 'MoE')
      edge('ffn_norm', 'tp_moe_in', [lbl(hidden(sym.sSp))])
      mIn = 'tp_moe_in'
    }
    const { T, dispatched, perExpert, localExperts } = moeTokens(cfg, par, train)
    const seqT = region ? sym.sCp : sym.sSp
    const Td = dm(T, `b·${seqT.sym}`)
    const E = dm(cfg.n_routed_experts, 'E')
    const K = dm(cfg.n_activated_experts, 'k')
    const tok = [Td, sym.d]
    const disp = [dm(dispatched, `${Td.sym}·k`), sym.d]
    const shared = cfg.n_shared_experts > 0
    const swiglu = (P) => pick(`${P}.w_gate.weight`, `${P}.w_gate.bias`, `${P}.w_up.weight`, `${P}.w_up.bias`, `${P}.w_down.weight`, `${P}.w_down.bias`)

    nextRow()
    node({ id: 'router', col: 0, kind: 'module', title: 'router', sub: tpOn ? 'replicated over TP' : 'gate', params: pick(`${M}.router.weight`, `${M}.router.bias`, `${M}.router.balance_bias`) })
    if (shared) {
      node({ id: 'shared', col: 2, kind: 'module', title: 'shared_experts', sub: `SwiGLU on every token${tpOn ? ' · TP-sharded' : ''}`, params: swiglu(`${M}.shared_experts`) })
    }
    edge(mIn, 'router', [lbl(tok, 'tokens')])
    if (shared) edge(mIn, 'shared')
    nextRow()
    node({ id: 'topk', col: 0, kind: 'op', title: 'top-k', sub: `sigmoid scores → ${cfg.n_activated_experts} of ${cfg.n_routed_experts}` })
    edge('router', 'topk', [lbl([Td, E], 'scores')])

    nextRow()
    const epGroup = `EP·${par.ep}`
    const epE = { axis: 'EP', group: epGroup, groupKey: 'ep', bytes: dispatched * D * cB }
    if (par.ep > 1) {
      node({
        id: 'dispatch',
        kind: 'comm',
        title: 'token dispatch',
        sub: 'expert parallel',
        fwd: [{ ...epE, op: 'all-to-all', note: `send each of the ${T.toLocaleString('en-US')} × ${cfg.n_activated_experts} token copies to the EP rank that owns its expert (${localExperts} experts per rank, balanced routing assumed)` }],
        bwd: [{ ...epE, op: 'all-to-all', note: 'return the grads of the dispatched tokens to the ranks they came from' }],
      })
    } else {
      node({ id: 'dispatch', kind: 'op', title: 'permute', sub: 'group token copies by expert' })
    }
    edge(mIn, 'dispatch')
    edge('topk', 'dispatch', [lbl([Td, K], 'indices')])

    nextRow()
    let expertFqns
    let expertTitle = 'experts'
    if (cfg.expert_layout === 'grouped') {
      expertFqns = pick(`${M}.experts.w_gate`, `${M}.experts.w_gate_bias`, `${M}.experts.w_up`, `${M}.experts.w_up_bias`, `${M}.experts.w_down`, `${M}.experts.w_down_bias`)
    } else {
      let e0 = 0
      for (let e = 0; e < cfg.n_routed_experts; e++) {
        const p = byFqn.get(`${M}.experts.${e}.w_gate.weight`)
        if (p && placeParam(p, par, coords).present) {
          e0 = e
          break
        }
      }
      expertFqns = swiglu(`${M}.experts.${e0}`)
      expertTitle = `experts.${e0}  (1 of ${localExperts} local)`
    }
    const perExp = Number.isInteger(perExpert) ? perExpert.toLocaleString('en-US') : perExpert.toFixed(1)
    node({ id: 'experts', kind: 'module', title: expertTitle, sub: `${localExperts} local experts × ~${perExp} tokens`, params: expertFqns })
    edge('dispatch', 'experts', [lbl(disp, par.ep > 1 ? 'recv' : 'routed')])

    nextRow()
    if (par.ep > 1) {
      node({
        id: 'combine',
        kind: 'comm',
        title: 'token combine',
        sub: 'expert parallel',
        fwd: [{ ...epE, op: 'all-to-all', note: 'return expert outputs to the ranks that own the tokens' }],
        bwd: [{ ...epE, op: 'all-to-all', note: 'send output grads back to the expert-owning ranks' }],
      })
    } else {
      node({ id: 'combine', kind: 'op', title: 'unpermute', sub: 'restore token order' })
    }
    edge('experts', 'combine', [lbl(disp)])

    nextRow()
    node({ id: 'wsum', kind: 'op', title: shared ? 'Σ gate · expert + shared' : 'Σ gate · expert', sub: 'weighted combine' })
    edge('combine', 'wsum', [lbl(disp)])
    edge('topk', 'wsum', [lbl([Td, K], 'gates')])
    if (shared) edge('shared', 'wsum', [lbl(tok)])
    ffnOut = 'wsum'
    ffnOutShape = hidden(seqT)
    exitNeeded = region
  }

  if (exitNeeded) {
    nextRow()
    tpExit('tp_ffn_out', moe ? 'MoE' : 'mlp')
    edge(ffnOut, 'tp_ffn_out', [lbl(ffnOutShape)])
    ffnOut = 'tp_ffn_out'
    ffnOutShape = hidden(sym.sSp)
  }
  nextRow()
  node({ id: 'add2', kind: 'add' })
  edge(ffnOut, 'add2', [lbl(ffnOutShape)])
  edges.push({ from: 'res2', to: 'add2', kind: 'residual', labels: [] })

  nextRow()
  node({ id: 'out', kind: 'io', title: `${L} output`, sub: li + 1 < cfg.n_layers ? `to layers.${li + 1}` : 'to final_norm' })
  if (bottom.fwd.length || bottom.bwd.length) {
    node({ id: 'dp_bottom', kind: 'comm', col: 2, title: `${L} params · ${par.dpStrategy.toUpperCase()}`, sub: 'parameter collectives', ...bottom })
  }
  edge('add2', 'out', [lbl(hidden(sym.sSp))])

  const totals = { fwd: 0, bwd: 0, fwdCalls: 0, bwdCalls: 0 }
  for (const n of nodes) {
    for (const dir of ['fwd', 'bwd']) {
      for (const e of n[dir]) {
        if (e.idle) continue
        totals[dir] += e.bytes * (e.count ?? 1)
        totals[`${dir}Calls`] += e.count ?? 1
      }
    }
  }
  return { nodes, edges, moe, totals, layerParams }
}
