// One transformer block as a data-flow graph on a single rank: modules with
// their local params, collectives inline where they fire, and edges labeled
// with the local tensor shapes. Comm nodes carry both forward and backward
// entries so the same layout renders either pass (backward = arrows reversed,
// conjugate collectives, gradient shapes).
//
// Nodes sit on a 3-lane grid (col 0 / 1 / 2, fractional allowed); `row` is the
// vertical order. Edges with route 'rail' run along a rail right of the grid
// (long skips from the attention input). Layout and drawing live in LayerView.jsx.
//
// The block is drawn as a generic `layers[i]`: parameter FQNs stay concrete
// (they name real tensors on this rank) but the structural labels do not, so the
// picture reads as any layer of the model. The ends of the model are always
// drawn around it -- input ids through tok_embeddings above, final_norm and the
// output projection below -- each with its own FSDP/HSDP unit, so one picture
// covers the whole forward path. Only the block's own collectives are summed
// into `totals`; the embedding and the head fire once per step, not per layer.

import { DTYPE_BYTES, prod } from './format.js'
import { compressRatio, hasIndexer, isHashLayer, isMoeLayer, layerMixer } from './model.js'
import { cpExchange, kvLocalHeads, linearState, moeTokens, moeTpRegion, seqDims } from './activations.js'
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
  const Li = 'layers[i]'
  // lm_head / tok_embeddings are only vocab-sharded when vocab parallelism is on
  const vp = tpOn && par.vocabParallel
  const moe = isMoeLayer(cfg, li)
  const mixer = layerMixer(cfg, li)
  const dsv4 = cfg.attn_type === 'dsv4'
  const hcM = cfg.hc_mult
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
  // DeepSeek-V4 hyper-connections carry hc_mult copies of the residual stream
  const stream = (seq) => (dsv4 ? [sym.b, seq, dm(hcM, 'm'), sym.d] : hidden(seq))
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

  // ---- Data-parallel param / grad collectives for one FSDP unit ---------------
  const sum = (ps, f) => ps.reduce((a, p) => a + f(p), 0)
  const full = (ps) => sum(ps, (p) => unshardedLocalNumel(p, par, coords))
  const local = (ps) => sum(ps, (p) => placeParam(p, par, coords).localNumel)
  const edpDeg = par.dp / par.ep

  const dpCollectives = (unitParams, unit) => {
    const trainable = unitParams.filter((p) => p.kind === 'param')
    const denseP = trainable.filter((p) => !p.expert)
    const expertP = trainable.filter((p) => p.expert)
    const top = { fwd: [], bwd: [] }
    const bottom = { fwd: [], bwd: [] }

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
        top.fwd.push({ ...e, op: 'all-gather', bytes: full(part.ps) * cB, note: `unshard ${unit} ${part.what} (${prec.compute}) just before its forward` })
        top.bwd.push({ ...e, op: 'reduce-scatter', bytes: full(part.ps) * gB, note: `average ${part.what} grads across the shard group and keep only this rank’s shard` })
        if (par.reshardAfterForward) {
          bottom.fwd.push({ ...e, op: 'reshard', idle: true, bytes: 0, note: `free the gathered params once forward leaves ${unit}` })
          bottom.bwd.push({ ...e, op: 'all-gather', bytes: full(part.ps) * cB, note: `re-unshard ${part.what} before ${unit}’s backward` })
        }
      }
      if (hsdp && par.dp / par.hsdpShard > 1) {
        const rep = par.dp / par.hsdpShard
        top.bwd.push({ axis, op: 'all-reduce', group: `replica·${rep}`, groupKey: null, bytes: local(trainable) * gB, note: 'average the sharded grads across HSDP replica blocks' })
      }
    }
    return { top, bottom }
  }

  const layerParams = params.filter((p) => p.layer === li)
  const { top, bottom } = dpCollectives(layerParams, Li)

  // A unit's gathers and reshards fold into one node: all-gather then reshard on
  // the way forward, re-gather then reduce-scatter on the way back.
  const dpNode = (id, dp, what) => {
    const fwd = [...dp.top.fwd, ...dp.bottom.fwd]
    const bwd = [...dp.bottom.bwd, ...dp.top.bwd]
    if (!fwd.length && !bwd.length) return
    node({ id, kind: 'comm', col: 2, title: `${what} · ${par.dpStrategy.toUpperCase()}`, sub: 'parameter / gradient collectives', fwd, bwd })
  }

  // ---- Embedding --------------------------------------------------------------
  nextRow()
  // int64 ids: the embedding lookup has no input grad, so backward stops here
  node({ id: 'tokens', kind: 'io', noGrad: true, title: 'input ids', sub: 'tokenized text · int64' })
  nextRow()
  node({
    id: 'tok_emb',
    kind: 'module',
    title: 'tok_embeddings',
    sub: vp ? 'vocab-parallel lookup' : 'embedding lookup',
    params: pick('tok_embeddings.weight'),
  })
  edge('tokens', 'tok_emb', [lbl([sym.b, sym.sCp], 'ids')], { noGrad: true })
  dpNode('dp_emb', dpCollectives(params.filter((p) => p.layer === null && p.group === 'embedding'), 'tok_embeddings'), 'embedding params')
  let embOut = 'tok_emb'
  if (vp) {
    // Each rank owns a vocab slice and zeroes the rows it does not hold, so the
    // partial lookups have to be summed (Megatron g, or reduce-scatter under SP).
    nextRow()
    tpExit('tp_emb_out', 'embedding')
    edge('tok_emb', 'tp_emb_out', [lbl(hidden(sym.sCp), 'partial h')])
    embOut = 'tp_emb_out'
  }
  if (dsv4) {
    nextRow()
    node({ id: 'hc_expand', kind: 'op', title: 'expand residual', sub: `1 → ${hcM} residual copies` })
    edge(embOut, 'hc_expand', [lbl(hidden(sym.sSp))])
    embOut = 'hc_expand'
  }

  // ---- Block input ------------------------------------------------------------
  const blockFrom = nodes.length
  nextRow()
  node({ id: 'in', kind: 'io', title: `${Li} input`, sub: `block i of ${cfg.n_layers}` })
  edge(embOut, 'in', [lbl(stream(sym.sSp))])
  if (top.fwd.length || top.bwd.length) {
    node({ id: 'dp_top', kind: 'comm', col: 2, title: `${Li} params · ${par.dpStrategy.toUpperCase()}`, sub: 'parameter / gradient collectives', ...top })
  }

  // ---- Attention --------------------------------------------------------------
  // Residual branch start: a fork, or hc_pre (hyper-connections), optionally
  // followed by an attention-residual mix over earlier layers (Kimi K3).
  const openSublayer = (id, hcKey, resKey, from, labels) => {
    nextRow()
    if (dsv4) {
      node({ id, kind: 'module', title: `hc_pre · ${hcKey === 'hc_attn' ? 'attention' : 'ffn'}`, sub: `Sinkhorn mix: ${hcM} residual copies → 1`, params: pick(`${L}.${hcKey}.fn`, `${L}.${hcKey}.base`, `${L}.${hcKey}.scale`) })
    } else {
      node({ id, kind: 'fork' })
    }
    edge(from, id, labels)
    if (!cfg.attn_res) return id
    nextRow()
    const resId = `${resKey}_res`
    node({ id: resId, kind: 'module', title: 'attention residual', sub: 'softmax-weighted mix of earlier layer outputs', params: pick(`${L}.${resKey}_res_norm.weight`, `${L}.${resKey}_res_proj.weight`) })
    edge(id, resId, dsv4 ? [lbl(hidden(sym.sSp))] : [])
    return resId
  }
  const closeSublayer = (id, resId, from, labels) => {
    nextRow()
    if (dsv4) node({ id, kind: 'op', title: 'hc_post', sub: `post ⊗ out + comb · residual → ${hcM} copies` })
    else node({ id, kind: 'add' })
    edge(from, id, labels)
    edges.push({ from: resId, to: id, kind: 'residual', labels: [] })
  }
  const cpNode = (id, col) => {
    const x = cpExchange(cfg, par, train, li, cB)
    const e = { axis: 'CP', group: `CP·${par.cp}`, groupKey: 'cp' }
    node({ id, col, kind: 'comm', title: x.title, sub: x.sub, fwd: x.fwd.map((v) => ({ ...e, ...v })), bwd: x.bwd.map((v) => ({ ...e, ...v })) })
  }

  const attnFrom = openSublayer('res1', 'hc_attn', 'attn', 'in', [lbl(stream(sym.sSp), 'x')])
  nextRow()
  node({ id: 'attn_norm', kind: 'module', title: 'attn_norm', sub: 'RMSNorm', params: pick(`${L}.attn_norm.weight`) })
  edge(attnFrom, 'attn_norm', dsv4 && !cfg.attn_res ? [lbl(hidden(sym.sSp))] : [])
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
  const colSub = tpOn ? 'column-parallel' : 'linear'

  // ---- Linear attention: Kimi Delta Attention / gated DeltaNet --------------
  const linearAttention = () => {
    const kda = mixer === 'kda'
    const st = linearState(cfg, par)
    const hv = dm(st.Hv, tpOn ? (kda ? 'h/tp' : 'hv/tp') : kda ? 'h' : 'hv')
    const hk = kda ? hv : dm(st.Hk, tpOn ? 'hk/tp' : 'hk')
    const dk = dm(st.dk, 'dk')
    const dv = dm(st.dv, 'dv')
    const vFlat = dm(st.Hv * st.dv, `${hv.sym}·dv`)
    nextRow()
    if (kda) {
      node({ id: 'la_qkv', col: 0, kind: 'module', title: 'q · k · v', sub: `${colSub} · short conv + SiLU`, params: pick(`${A}.q_proj.weight`, `${A}.k_proj.weight`, `${A}.v_proj.weight`, `${A}.q_conv1d.weight`, `${A}.k_conv1d.weight`, `${A}.v_conv1d.weight`) })
      node({ id: 'la_gates', col: 2, kind: 'module', title: 'decay · write · output gates', sub: 'α low-rank · β = σ(b) · g full-rank', params: pick(`${A}.f_a_proj.weight`, `${A}.f_b_proj.weight`, `${A}.A_log`, `${A}.dt_bias`, `${A}.b_proj.weight`, `${A}.g_proj.weight`) })
    } else {
      node({ id: 'la_qkv', col: 0, kind: 'module', title: 'in_proj_qkv', sub: `${colSub} · short conv + SiLU`, params: pick(`${A}.in_proj_qkv.weight`, `${A}.conv1d.weight`) })
      node({ id: 'la_gates', col: 2, kind: 'module', title: 'decay · write · output gates', sub: 'α from a · β = σ(b) · z gate', params: pick(`${A}.in_proj_a.weight`, `${A}.in_proj_b.weight`, `${A}.A_log`, `${A}.dt_bias`, `${A}.in_proj_z.weight`) })
    }
    edge(attnIn, 'la_qkv', inLbl)
    edge(attnIn, 'la_gates')
    if (par.cp > 1) {
      nextRow()
      cpNode('cp', 1)
    }
    nextRow()
    node({ id: 'la_rule', kind: 'op', title: kda ? 'Kimi delta rule' : 'gated delta rule', sub: 'chunked · S ← α S + β (v − S k) kᵀ' })
    edge('la_qkv', 'la_rule', [lbl([sym.b, sym.sCp, hk, dk], 'q,k'), lbl([sym.b, sym.sCp, hv, dv], 'v')])
    edge('la_gates', 'la_rule', [lbl([sym.b, sym.sCp, kda ? dm(st.Hv * st.dk, `${hv.sym}·dk`) : hv], 'α'), lbl([sym.b, sym.sCp, hv], 'β')])
    if (par.cp > 1) edge('cp', 'la_rule', [lbl([sym.b, hv, dk, dv], 'S')])
    nextRow()
    node({ id: 'la_norm', kind: 'module', title: 'gated RMSNorm', sub: kda ? 'norm(o) ⊙ σ(g)' : 'norm(o) ⊙ SiLU(z)', params: pick(kda ? `${A}.o_norm.weight` : `${A}.norm.weight`) })
    edge('la_rule', 'la_norm', [lbl([sym.b, sym.sCp, hv, dv], 'o')])
    edge('la_gates', 'la_norm', [lbl([sym.b, sym.sCp, vFlat], kda ? 'g' : 'z')])
    nextRow()
    node({ id: 'wo', kind: 'module', title: kda ? 'o_proj' : 'out_proj', sub: tpOn ? 'row-parallel' : 'output proj', params: pick(kda ? `${A}.o_proj.weight` : `${A}.out_proj.weight`) })
    edge('la_norm', 'wo', [lbl([sym.b, sym.sCp, vFlat])])
    return 'wo'
  }

  // ---- DeepSeek-V4: low-rank q, MQA window KV, compressor, indexer ----------
  const v4Attention = () => {
    const r = compressRatio(cfg, li)
    const indexer = r === 4 && cfg.index_n_heads > 0
    const hd = dm(cfg.head_dim, 'hd')
    const qr = dm(cfg.q_lora_rank, 'q_rank')
    const W = cfg.window_size
    const Sq = sym.sCp
    const qFlat = dm(H.n * cfg.head_dim, `${H.sym}·hd`)
    nextRow() // A
    node({ id: 'wq_a', col: 0, kind: 'module', title: 'wq_a', sub: 'low-rank down · replicated', params: pick(`${A}.wq_a.weight`) })
    node({ id: 'wkv', col: 1, kind: 'module', title: 'wkv', sub: 'one KV head (MQA) · replicated', params: pick(`${A}.wkv.weight`) })
    edge(attnIn, 'wq_a')
    edge(attnIn, 'wkv', inLbl)
    nextRow() // B
    node({ id: 'q_norm', col: 0, kind: 'module', title: 'q_norm', sub: 'RMSNorm', params: pick(`${A}.q_norm.weight`) })
    node({ id: 'kv_norm', col: 1, kind: 'module', title: 'kv_norm', sub: 'RMSNorm + RoPE', params: pick(`${A}.kv_norm.weight`) })
    edge('wq_a', 'q_norm', [lbl([sym.b, Sq, qr])])
    edge('wkv', 'kv_norm', [lbl([sym.b, Sq, hd])])
    nextRow() // C
    node({ id: 'wq_b', col: 0, kind: 'module', title: 'wq_b', sub: tpOn ? 'column-parallel' : 'up proj', params: pick(`${A}.wq_b.weight`) })
    edge('q_norm', 'wq_b', [lbl([sym.b, Sq, qr])])
    let kvSrc = 'kv_norm'
    let kvLen = Sq
    if (r) {
      node({ id: 'compressor', col: 1, kind: 'module', title: `KV compressor ×${r}`, sub: `${r === 4 ? 'overlapping ' : ''}gated pooling · after window KV`, params: pick(`${A}.compressor.ape`, `${A}.compressor.wkv.weight`, `${A}.compressor.wgate.weight`, `${A}.compressor.norm.weight`) })
      edge('kv_norm', 'compressor', [lbl([sym.b, Sq, hd], 'window kv')])
      edge(attnIn, 'compressor', [], { route: 'rail' })
      kvSrc = 'compressor'
      kvLen = dm(sCp + cdiv(sCp, r), `${Sq.sym}+${Sq.sym}/${r}`)
    }
    nextRow() // D
    node({ id: 'q_head', col: 0, kind: 'op', title: 'head RMSNorm + RoPE', sub: 'per-head norm on q' })
    edge('wq_b', 'q_head', [lbl([sym.b, Sq, qFlat])])
    const topk = Math.min(cfg.index_topk, cdiv(S, 4))
    if (indexer) {
      node({
        id: 'indexer',
        col: 2,
        kind: 'module',
        title: 'lightning indexer',
        sub: `${cfg.index_n_heads} heads${tpOn ? ' · TP-sharded' : ''} · own ×4 compressor`,
        params: pick(`${A}.indexer.wq_b.weight`, `${A}.indexer.weights_proj.weight`, `${A}.indexer.compressor.ape`, `${A}.indexer.compressor.wkv.weight`, `${A}.indexer.compressor.wgate.weight`, `${A}.indexer.compressor.norm.weight`),
      })
      edge('q_norm', 'indexer', [lbl([sym.b, Sq, qr], 'qr')], { route: 'early', labelAt: 'mid' })
      edge(attnIn, 'indexer', [], { route: 'rail' })
    }
    let kvIn = kvSrc
    let idxSrc = indexer ? 'indexer' : null
    if (par.cp > 1 || (indexer && tpOn)) {
      nextRow() // E
      if (par.cp > 1) {
        cpNode('cp', 1)
        // the TP score all-reduce shares this gap; cp → attention still shows the kv shape
        edge(kvSrc, 'cp', indexer && tpOn ? [] : [lbl([sym.b, kvLen, hd], 'kv')])
        kvIn = 'cp'
      }
      if (indexer && tpOn) {
        const bytes = b * sCp * cdiv(S, 4) * 4
        node({
          id: 'idx_ar',
          col: 2,
          kind: 'comm',
          title: 'index scores',
          sub: 'indexer heads are TP-sharded',
          fwd: [tpEntry('all-reduce', bytes, 'sum the index scores [b, s, s/4] (fp32) from each TP rank’s share of indexer heads before picking top-k')],
          bwd: [tpEntry('identity', bytes, 'the backward of a sum passes the gradient through unchanged', true)],
        })
        edge('indexer', 'idx_ar', [lbl([sym.b, Sq, dm(cdiv(S, 4), 's/4')], 'scores')])
        idxSrc = 'idx_ar'
      }
    }
    nextRow() // F
    node({
      id: 'sdpa',
      kind: 'op',
      title: 'sparse attention',
      sub: indexer ? `window ${W} + top-${topk} compressed blocks` : r ? `window ${W} + all ×${r} compressed blocks` : `sliding window ${W}`,
      params: pick(`${A}.attn_sink`),
    })
    edge('q_head', 'sdpa', [lbl([sym.b, Sq, H, hd], 'q')])
    const kvOut = par.cp > 1 && r ? dm(sCp + cdiv(S, r), `${Sq.sym}+s/${r}`) : kvLen
    edge(kvIn, 'sdpa', [lbl([sym.b, kvOut, hd], 'kv')])
    // with both CP and the TP score all-reduce the label would collide with kv's; the node subtitle carries top-k
    if (idxSrc) edge(idxSrc, 'sdpa', par.cp > 1 && idxSrc === 'idx_ar' ? [] : [lbl([sym.b, Sq, dm(topk, 'topk')], 'top-k idx')])
    nextRow() // G
    node({ id: 'wo_a', kind: 'module', title: 'wo_a', sub: `grouped low-rank · ${cfg.o_groups} groups${tpOn ? ' · col-parallel' : ''}`, params: pick(`${A}.wo_a.weight`) })
    edge('sdpa', 'wo_a', [lbl([sym.b, Sq, qFlat])])
    nextRow() // H
    node({ id: 'wo', kind: 'module', title: 'wo_b', sub: tpOn ? 'row-parallel' : 'output proj', params: pick(`${A}.wo_b.weight`) })
    edge('wo_a', 'wo', [lbl([sym.b, Sq, dm(cdiv(cfg.o_groups, tp) * cfg.o_lora_rank, tpOn ? 'g/tp·o_rank' : 'g·o_rank')])])
    return 'wo'
  }

  // ---- Softmax attention: MHA / GQA / MLA -----------------------------------
  const softmaxAttention = () => {
  let qOut
  let qLabels
  let kvSources
  let indexerOut = false
  const gate = cfg.attn_output_gate

  if (cfg.attn_type !== 'mla') {
    const hd = dm(cfg.head_dim, 'hd')
    const KVn = cfg.attn_type === 'mha' ? cfg.n_heads : cfg.n_kv_heads
    const KVsym = cfg.attn_type === 'mha' ? 'h' : 'kv'
    const kvRep = tpOn && KVn % tp !== 0
    const kvh = dm(kvLocalHeads(cfg, par), tpOn ? (kvRep ? '1' : `${KVsym}/tp`) : KVsym)
    const qShape = [sym.b, sym.sCp, H, hd]
    const kShape = [sym.b, sym.sCp, kvh, hd]
    nextRow()
    node({ id: 'wq', col: 0, kind: 'module', title: 'wq', sub: gate ? `${colSub} · q + output gate` : colSub, params: pick(`${A}.wq.weight`, `${A}.wq.bias`) })
    node({ id: 'wk', col: 1, kind: 'module', title: 'wk', sub: kvRep ? 'KV heads replicated over TP' : colSub, params: pick(`${A}.wk.weight`, `${A}.wk.bias`) })
    node({ id: 'wv', col: 2, kind: 'module', title: 'wv', sub: kvRep ? 'KV heads replicated over TP' : colSub, params: pick(`${A}.wv.weight`, `${A}.wv.bias`) })
    edge(attnIn, 'wq')
    edge(attnIn, 'wk', inLbl)
    edge(attnIn, 'wv')
    let qPrev = 'wq'
    let kPrev = 'wk'
    if (cfg.qk_norm) {
      nextRow()
      const normSub = cfg.qk_norm_type === 'full' ? `RMSNorm over all heads${tpOn ? ' · sharded' : ''}` : 'RMSNorm per head'
      node({ id: 'q_norm', col: 0, kind: 'module', title: 'q_norm', sub: normSub, params: pick(`${A}.q_norm.weight`) })
      node({ id: 'k_norm', col: 1, kind: 'module', title: 'k_norm', sub: normSub, params: pick(`${A}.k_norm.weight`) })
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
  } else {
    const qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    const qkd = dm(qk, 'qk_hd')
    const vd = dm(cfg.v_head_dim, 'v_hd')
    const ropeD = dm(cfg.qk_rope_head_dim, 'rope_hd')
    const kvr = dm(cfg.kv_lora_rank, 'kv_rank')
    const one = dm(1, '1')
    const lora = cfg.q_lora_rank > 0
    const indexer = hasIndexer(cfg, li)
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
    // With an indexer, lane 2 carries the indexer and k_rope's RoPE folds into the concat node.
    if (!indexer) node({ id: 'rope_k', col: 2, kind: 'op', title: 'RoPE', sub: 'shared k_rope (1 head)' })
    if (lora) edge('q_a_norm', 'q_b', [lbl([sym.b, sym.sCp, qr])])
    if (indexer) {
      node({
        id: 'indexer',
        col: 2,
        kind: 'module',
        title: 'lightning indexer',
        sub: `DSA · ${cfg.index_n_heads} heads · replicated over TP`,
        params: pick(`${A}.indexer.wq_b.weight`, `${A}.indexer.wk.weight`, `${A}.indexer.k_norm.weight`, `${A}.indexer.k_norm.bias`, `${A}.indexer.weights_proj.weight`),
      })
      edge(attnIn, 'indexer', [], { route: 'early' })
      if (lora) edge('q_a_norm', 'indexer', [lbl([sym.b, sym.sCp, qr], 'c_q')], { route: 'early', labelAt: 'mid' })
    }
    edge('kv_split', 'kv_a_norm', [lbl([sym.b, sym.sCp, kvr], 'c_kv')])
    if (!indexer) edge('kv_split', 'rope_k', [lbl([sym.b, sym.sCp, one, ropeD], 'k_rope')], { labelAt: 'mid' })
    nextRow() // D
    node({ id: 'rope_q', col: 0, kind: 'op', title: 'split + RoPE', sub: 'q_nope | RoPE(q_rope)' })
    node({ id: 'kv_b', col: 1, kind: 'module', title: 'kv_b_proj', sub: tpOn ? 'column-parallel' : 'up proj', params: pick(`${A}.kv_b_proj.weight`) })
    edge(lora ? 'q_b' : 'q_a', 'rope_q', [lbl(qFlat)])
    edge('kv_a_norm', 'kv_b', [lbl([sym.b, sym.sCp, kvr])])
    nextRow() // E
    node({ id: 'kv_join', col: 1, kind: 'op', title: 'split + concat', sub: indexer ? 'k = [k_nope, RoPE(k_rope)], v' : 'k = [k_nope, k_rope], v' })
    edge('kv_b', 'kv_join', [lbl([sym.b, sym.sCp, dm(H.n * (cfg.qk_nope_head_dim + cfg.v_head_dim), `${H.sym}·(nope+v_hd)`)])])
    if (!indexer) edge('rope_k', 'kv_join')
    qOut = 'rope_q'
    qLabels = [lbl([sym.b, sym.sCp, H, qkd], 'q')]
    kvSources = [['kv_join', [lbl([sym.b, sym.sCp, H, qkd], 'k'), lbl([sym.b, sym.sCp, H, vd], 'v')]]]
    if (indexer) indexerOut = true
  }

  let sdpaIn = kvSources
  const ring = par.cpStyle === 'ring'
  if (par.cp > 1) {
    nextRow()
    cpNode('cp', kvSources.length === 2 ? 1.5 : 1)
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
    sub: indexerOut
      ? `sparse · each query reads top-${Math.min(cfg.index_topk || S, S).toLocaleString('en-US')} keys`
      : `${par.cp > 1 ? (ring ? 'causal · blockwise over the ring' : 'causal · local q vs all k,v') : 'causal · flash'}${gate ? ' · σ-gated output' : ''}`,
    params: pick(`${A}.sinks`),
  })
  const mlaGate = gate && cfg.attn_type === 'mla'
  if (mlaGate) {
    node({ id: 'g_proj', col: 2, kind: 'module', title: 'g_proj', sub: `output gate${tpOn ? ' · column-parallel' : ''}`, params: pick(`${A}.g_proj.weight`) })
    edge(attnIn, 'g_proj', [], { route: 'rail' })
  }
  edge(qOut, 'sdpa', qLabels)
  if (indexerOut) {
    const topk = Math.min(cfg.index_topk || S, S)
    edge('indexer', 'sdpa', [lbl([sym.b, sym.sCp, dm(topk, 'topk')], 'top-k idx')])
  }
  for (const [src, labels] of sdpaIn) edge(src, 'sdpa', labels)
  const outDim = cfg.attn_type === 'mla' ? dm(H.n * cfg.v_head_dim, `${H.sym}·v_hd`) : dm(H.n * cfg.head_dim, `${H.sym}·hd`)
  nextRow()
  node({ id: 'wo', kind: 'module', title: 'wo', sub: tpOn ? 'row-parallel' : 'output proj', params: pick(`${A}.wo.weight`, `${A}.wo.bias`) })
  edge('sdpa', 'wo', [lbl([sym.b, sym.sCp, outDim])])
  if (mlaGate) edge('g_proj', 'wo', [lbl([sym.b, sym.sCp, outDim], 'σ(g)')])
  return 'wo'
  }

  const mixOut = mixer === 'kda' || mixer === 'gdn' ? linearAttention() : mixer === 'dsv4' ? v4Attention() : softmaxAttention()
  let attnOut = mixOut
  if (tpOn) {
    nextRow()
    tpExit('tp_attn_out', 'attention')
    edge(mixOut, 'tp_attn_out', [lbl(hidden(sym.sCp))])
    attnOut = 'tp_attn_out'
  }
  closeSublayer('add1', 'res1', attnOut, [lbl(hidden(sym.sSp))])

  // ---- FFN ----------------------------------------------------------------------
  const ffnFrom = openSublayer('res2', 'hc_ffn', 'mlp', 'add1', dsv4 ? [lbl(stream(sym.sSp))] : [])
  let ffnOut
  let ffnOutShape
  let exitNeeded

  if (!moe) {
    const F = `${L}.mlp`
    nextRow()
    node({ id: 'ffn_norm', kind: 'module', title: 'mlp_norm', sub: 'RMSNorm', params: pick(`${L}.mlp_norm.weight`) })
    edge(ffnFrom, 'ffn_norm', dsv4 && !cfg.attn_res ? [lbl(hidden(sym.sSp))] : [])
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
    edge(ffnFrom, 'ffn_norm', dsv4 && !cfg.attn_res ? [lbl(hidden(sym.sSp))] : [])
    const region = moeTpRegion(cfg, par)
    let mIn = 'ffn_norm'
    if (region) {
      nextRow()
      tpEnter('tp_moe_in', 'MoE')
      edge('ffn_norm', 'tp_moe_in', [lbl(hidden(sym.sSp))])
      mIn = 'tp_moe_in'
    }
    const { T, dispatched, perExpert, localExperts, width } = moeTokens(cfg, par, train)
    const latent = cfg.moe_latent_dim > 0
    const hash = isHashLayer(cfg, li)
    const seqT = region ? sym.sCp : sym.sSp
    const Td = dm(T, `b·${seqT.sym}`)
    const E = dm(cfg.n_routed_experts, 'E')
    const K = dm(cfg.n_activated_experts, 'k')
    const tok = [Td, sym.d]
    const dx = latent ? dm(width, 'd_lat') : sym.d
    const disp = [dm(dispatched, `${Td.sym}·k`), dx]
    const shared = cfg.n_shared_experts > 0
    const swiglu = (P) => pick(`${P}.w_gate.weight`, `${P}.w_gate.bias`, `${P}.w_up.weight`, `${P}.w_up.bias`, `${P}.w_down.weight`, `${P}.w_down.bias`)

    nextRow()
    node({ id: 'router', col: 0, kind: 'module', title: 'router', sub: hash ? 'hash routing: token id → experts' : tpOn ? 'replicated over TP' : 'gate', params: pick(`${M}.router.weight`, `${M}.router.bias`, `${M}.router.balance_bias`, `${M}.router.tid2eid`) })
    if (shared) {
      node({ id: 'shared', col: 2, kind: 'module', title: 'shared_experts', sub: `${cfg.shared_expert_gate ? 'σ-gated ' : ''}SwiGLU on every token${tpOn ? ' · TP-sharded' : ''}`, params: [...swiglu(`${M}.shared_experts`), ...pick(`${M}.shared_gate.weight`)] })
    }
    edge(mIn, 'router', [lbl(tok, 'tokens')])
    if (shared) edge(mIn, 'shared')
    nextRow()
    node({ id: 'topk', col: 0, kind: 'op', title: hash ? 'hash lookup' : 'top-k', sub: hash ? `expert ids = tid2eid[token] (${cfg.n_activated_experts})` : `scores → ${cfg.n_activated_experts} of ${cfg.n_routed_experts}` })
    edge('router', 'topk', [lbl([Td, E], 'scores')])
    if (latent) {
      node({ id: 'latent_down', col: 1, kind: 'module', title: 'latent_down', sub: `experts run at width ${width.toLocaleString('en-US')}`, params: pick(`${M}.latent_down.weight`) })
    }

    nextRow()
    const epGroup = `EP·${par.ep}`
    const epE = { axis: 'EP', group: epGroup, groupKey: 'ep', bytes: dispatched * width * cB }
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
    if (latent) {
      edge(mIn, 'latent_down')
      edge('latent_down', 'dispatch', [lbl([Td, dx])])
    } else {
      edge(mIn, 'dispatch')
    }
    edge('topk', 'dispatch', [lbl([Td, K], 'indices')], { noGrad: true })

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
    const sumHere = shared && !latent
    node({ id: 'wsum', kind: 'op', title: sumHere ? 'Σ gate · expert + shared' : 'Σ gate · expert', sub: 'weighted combine' })
    edge('combine', 'wsum', [lbl(disp)])
    edge('topk', 'wsum', [lbl([Td, K], 'gates')])
    ffnOut = 'wsum'
    if (latent) {
      nextRow()
      node({ id: 'latent_up', kind: 'module', title: 'latent_norm + latent_up', sub: 'back to model width', params: pick(`${M}.latent_norm.weight`, `${M}.latent_up.weight`) })
      edge('wsum', 'latent_up', [lbl([Td, dx])])
      ffnOut = 'latent_up'
      if (shared) {
        nextRow()
        node({ id: 'add_shared', kind: 'op', title: '+ shared experts', sub: 'routed + shared' })
        edge('latent_up', 'add_shared', [lbl(tok)])
        ffnOut = 'add_shared'
      }
    }
    if (shared) edge('shared', ffnOut, [lbl(tok)])
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
  closeSublayer('add2', 'res2', ffnOut, [lbl(ffnOutShape)])

  nextRow()
  node({ id: 'out', kind: 'io', title: `${Li} output`, sub: 'to layers[i+1], or the head' })
  if (bottom.fwd.length || bottom.bwd.length) {
    node({ id: 'dp_bottom', kind: 'comm', col: 2, title: `${Li} params · ${par.dpStrategy.toUpperCase()}`, sub: 'parameter collectives', ...bottom })
  }
  edge('add2', 'out', [lbl(stream(sym.sSp))])
  const blockTo = nodes.length

  // ---- Model head -------------------------------------------------------------
  {
    let src = 'out'
    let shape = stream(sym.sSp)
    const step = (id, n, labels = [lbl(shape)]) => {
      nextRow()
      node({ id, ...n })
      edge(src, id, labels)
      src = id
    }
    if (dsv4) {
      step('hc_head', { kind: 'module', title: 'hc_head', sub: `Sinkhorn mix: ${hcM} residual copies → 1`, params: pick('hc_head.fn', 'hc_head.base', 'hc_head.scale') })
      shape = hidden(sym.sSp)
    }
    if (cfg.attn_res) {
      step('out_res', { kind: 'module', title: 'output residual', sub: 'softmax-weighted mix of earlier layer outputs', params: pick('output_res_norm.weight', 'output_res_proj.weight') })
    }
    step('final_norm', { kind: 'module', title: 'final_norm', sub: 'RMSNorm', params: pick('final_norm.weight') })
    dpNode('dp_head', dpCollectives(params.filter((p) => p.layer === null && p.group !== 'embedding'), 'the head'), 'head params')

    // lm_head is vocab-parallel only when asked for; otherwise it is replicated
    // and each rank keeps the logits for its own sequence shard (no collective).
    let seq = sym.sSp
    if (vp) {
      nextRow()
      tpEnter('tp_head_in', 'lm_head')
      edge(src, 'tp_head_in', [lbl(hidden(sym.sSp))])
      src = 'tp_head_in'
      seq = sym.sCp
    }
    shape = hidden(seq)
    const tied = cfg.tie_embeddings
    step('lm_head', {
      kind: 'module',
      title: 'lm_head',
      sub: `${vp ? 'vocab-parallel' : 'd → vocab'}${tied ? ' · tied embeddings' : ''}`,
      params: pick(tied ? 'tok_embeddings.weight' : 'lm_head.weight'),
    })
    const V = dm(vp ? cdiv(cfg.vocab_size, tp) : cfg.vocab_size, vp ? 'V/tp' : 'V')
    step('logits', { kind: 'io', title: 'logits', sub: vp ? 'vocab-parallel → loss parallel' : 'to cross-entropy loss' }, [lbl([sym.b, seq, V])])
  }

  // Only the block itself: the embedding and the head fire once per step, so
  // folding them in would make the per-layer numbers wrong.
  const totals = { fwd: 0, bwd: 0, fwdCalls: 0, bwdCalls: 0 }
  for (const n of nodes.slice(blockFrom, blockTo)) {
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
