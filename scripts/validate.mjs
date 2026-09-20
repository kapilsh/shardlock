// Invariant checks for the sharding / memory math — run with `npm run validate`.
//
//  1. Reference parameter counts match known models.
//  2. For every param under many parallel configs: the distinct shards held by
//     all ranks tile the global tensor exactly (in-bounds, total numel matches),
//     and every shard is held by the same number of ranks.
//  3. Summed parameter bytes over the world == global bytes × replication.

import { PRESETS, fullModelStats } from '../src/lib/presets.js'
import { buildModel, validateModel } from '../src/lib/model.js'
import { rankCoords, validateParallel, worldSize, DEFAULT_PARALLEL } from '../src/lib/mesh.js'
import { placeParam, shardLevels, tpApplies } from '../src/lib/shard.js'
import { prod } from '../src/lib/format.js'
import { stepCollectives } from '../src/lib/comms.js'
import { activationSections } from '../src/lib/activations.js'
import { worldMemory, DEFAULT_PRECISION } from '../src/lib/memory.js'
import { buildLayerFlow } from '../src/lib/layerFlow.js'

let failures = 0
const fail = (msg) => {
  failures++
  if (failures <= 25) console.error('  ✗ ' + msg)
}
const eq = (a, b, msg) => a !== b && fail(`${msg}: got ${a}, expected ${b}`)

const total = (params, filter = () => true) =>
  params.filter((p) => p.kind === 'param' && filter(p)).reduce((a, p) => a + prod(p.shape), 0)

// ---- 1. reference counts ----------------------------------------------------
{
  const tiny = buildModel(PRESETS['tiny-moe'].config)
  eq(total(tiny), 4_396_288, 'tiny-moe total params (matches moe_model.MoEDecoder)')

  const llama = buildModel(PRESETS['llama3-8b'].config)
  eq(total(llama, (p) => p.layer === 0), 218_112_000, 'llama3-8b params per layer')
  eq(total(llama, (p) => p.fqn === 'tok_embeddings.weight'), 525_336_576, 'llama3-8b embedding')

  const ds = buildModel(PRESETS['deepseek-v3'].config)
  eq(total(ds, (p) => p.layer === 1 && p.group === 'attention'), 187_105_280, 'deepseek-v3 MLA attention params (excl. norms)')
  eq(total(ds, (p) => p.layer === 1 && p.group === 'experts'), 256 * 3 * 7168 * 2048, 'deepseek-v3 routed expert params')
  console.log('✓ reference parameter counts')
}

// ---- 1b. presets at full depth vs Hugging Face safetensors totals -----------
{
  let n = 0
  for (const [key, preset] of Object.entries(PRESETS)) {
    const st = fullModelStats(preset)
    if (!st || !preset.published) continue
    // Llama 4 checkpoints also carry the vision encoder, which is not modeled.
    const tol = preset.publishedExtra === 'vision' ? 0.01 : 1e-4
    const rel = Math.abs(st.total - preset.published) / preset.published
    if (rel > tol) fail(`${key}: full-depth params ${st.total} vs published ${preset.published} (${(rel * 100).toFixed(3)}%)`)
    n++
  }
  console.log(`✓ ${n} presets match published parameter totals at full depth`)
}

// ---- 2/3. tiling invariants -------------------------------------------------
const pars = []
for (const dpStrategy of ['ddp', 'fsdp', 'hsdp'])
  for (const dp of [1, 2, 4, 8])
    for (const tp of [1, 2, 4, 8])
      for (const cp of [1, 2])
        for (const ep of [1, 2, 4])
          for (const hsdpShard of [2, 4])
            pars.push({ ...DEFAULT_PARALLEL, dpStrategy, dp, tp, cp, ep, hsdpShard })

let checked = 0
for (const key of Object.keys(PRESETS)) {
  for (const layout of ['grouped', 'module_list']) {
    const cfg = { ...PRESETS[key].config, expert_layout: layout, vocab_size: PRESETS[key].config.vocab_size + 3 }
    if (cfg.arch === 'dense' && layout === 'module_list') continue
    if (validateModel(cfg).errors.length) fail(`${key}: preset invalid`)
    // Keep the per-expert module list small so the check stays fast.
    if (layout === 'module_list') cfg.n_routed_experts = Math.min(cfg.n_routed_experts, 16)
    const params = buildModel(cfg)
    for (const par of pars) {
      for (const tpExperts of [true, false]) {
        const p2 = { ...par, tpExperts }
        if (validateParallel(p2, cfg).errors.length) continue
        checked++
        const W = worldSize(p2)
        const coords = Array.from({ length: W }, (_, r) => rankCoords(p2, r))
        for (const p of params) {
          const levels = shardLevels(p, p2)
          const cells = new Map()
          for (const c of coords) {
            const pl = placeParam(p, p2, c, levels)
            if (!pl.present) continue
            pl.ranges.forEach((r, d) => {
              if (r.start < 0 || r.start + r.len > p.shape[d]) fail(`${p.fqn} out of bounds on dim ${d}`)
            })
            const k = pl.ranges.map((r) => `${r.start}+${r.len}`).join('|')
            cells.set(k, (cells.get(k) ?? 0) + 1)
          }
          const tiled = [...cells.keys()].reduce(
            (a, k) => a + k.split('|').reduce((m, x) => m * Number(x.split('+')[1]), 1),
            0,
          )
          const ctx = `${key}/${layout} ${p2.dpStrategy} dp${p2.dp} tp${p2.tp} cp${p2.cp} ep${p2.ep} h${p2.hsdpShard} tpE${tpExperts} ${p.fqn}`
          eq(tiled, prod(p.shape), `${ctx}: shards do not tile the tensor`)
          const holders = new Set([...cells.entries()].filter(([k]) => !/\+0(\||$)/.test(k)).map(([, v]) => v))
          if (holders.size > 1) fail(`${ctx}: uneven replication ${[...holders]}`)
        }
      }
    }
    // smoke: other views don't throw on a representative config
    const par = { ...DEFAULT_PARALLEL, dp: 4, tp: 2, cp: 2, ep: cfg.arch === 'moe' ? 2 : 1 }
    const train = { micro_batch: 2, seq_len: 4096 }
    if (!validateParallel(par, cfg, train).errors.length) {
      activationSections(cfg, par, train, DEFAULT_PRECISION)
      stepCollectives(cfg, par, train, DEFAULT_PRECISION, params)
      for (const p3 of [par, { ...par, dpStrategy: 'hsdp', hsdpShard: 2 }, { ...par, dpStrategy: 'ddp', cpStyle: 'allgather', sp: false }]) {
        for (let li = 0; li < cfg.n_layers; li++) {
          const g = buildLayerFlow(cfg, p3, train, DEFAULT_PRECISION, params, li, rankCoords(p3, 0))
          const ids = new Set(g.nodes.map((n) => n.id))
          if (ids.size !== g.nodes.length) fail(`${key}/${layout} layer ${li}: duplicate node ids`)
          for (const e of g.edges) if (!ids.has(e.from) || !ids.has(e.to)) fail(`${key}/${layout} layer ${li}: dangling edge ${e.from}→${e.to}`)
          const shown = new Set(g.nodes.flatMap((n) => n.params ?? []))
          for (const p of params.filter((q) => q.layer === li && !(q.expert && q.expert.index != null))) {
            if (!shown.has(p.fqn)) fail(`${key}/${layout} layer ${li}: ${p.fqn} missing from layer flow`)
          }
          // every layer is drawn between the embedding and the model head
          for (const p of params.filter((q) => q.layer === null)) {
            if (!shown.has(p.fqn)) fail(`${key}/${layout} layer ${li}: root param ${p.fqn} missing from layer flow`)
          }
          const lm = cfg.tie_embeddings ? 'tok_embeddings.weight' : 'lm_head.weight'
          if (!shown.has(lm)) fail(`${key}/${layout} layer ${li}: ${lm} missing from layer flow head`)
        }
      }
      const wm = worldMemory(params, par, DEFAULT_PRECISION)
      if (!(wm.maxPeak > 0)) fail(`${key}: memory not positive`)
    }
  }
}
console.log(`✓ tiling checked across ${checked} (model × parallel) configs`)

// ---- 3. FSDP bytes conservation ---------------------------------------------
{
  const cfg = PRESETS['tiny-moe'].config
  const params = buildModel(cfg)
  const par = { ...DEFAULT_PARALLEL, dpStrategy: 'fsdp', dp: 8, tp: 2, cp: 1, ep: 4 }
  const prec = { ...DEFAULT_PRECISION, optimizer: 'none', grad: 'fp32' }
  const wm = worldMemory(params, par, prec)
  const sum = wm.rows.reduce((a, r) => a + r.params, 0)
  // TP-replicated params (norms, router) live on every TP rank.
  const expected = params
    .filter((p) => p.kind === 'param')
    .reduce((a, p) => a + prod(p.shape) * (tpApplies(p, par) ? 1 : par.tp), 0)
  eq(sum, expected * 4, 'FSDP+TP+EP: world param bytes == global bytes × TP replication')
  console.log('✓ FSDP byte conservation')
}

if (failures) {
  console.error(`\n✗ ${failures} failure(s)`)
  process.exit(1)
}
console.log('\n✓ All sharding invariants hold.')
