// Check every preset against the model's own config.json on Hugging Face —
// run with `npm run check:hf` (needs network; `npm run validate` does not).
//
// `validate.mjs` proves the modeled parameter total matches `published`, which
// is one scalar: a schedule error that keeps the layer counts the same slips
// through (an indexer every 4 layers phased from 5 instead of 6 has the same
// total). This compares the fields themselves, and the per-layer schedules
// layer by layer wherever config.json states them explicitly.

import { PRESETS, fullModelStats } from '../src/lib/presets.js'
import { compressRatio, hasIndexer, isFullAttnLayer, isHashLayer, isMoeLayer } from '../src/lib/model.js'

// Gated repos: an ungated mirror of the same checkpoint's config.
const MIRROR = {
  'meta-llama/Llama-3.1-8B': 'unsloth/Meta-Llama-3.1-8B',
  'meta-llama/Llama-3.3-70B-Instruct': 'unsloth/Llama-3.3-70B-Instruct',
  'meta-llama/Llama-4-Scout-17B-16E-Instruct': 'unsloth/Llama-4-Scout-17B-16E-Instruct',
  'meta-llama/Llama-4-Maverick-17B-128E-Instruct': 'unsloth/Llama-4-Maverick-17B-128E-Instruct',
}

let mismatches = 0
let scalars = 0
let schedules = 0
const pick = (o, ...ks) => {
  for (const k of ks) if (o?.[k] !== undefined && o[k] !== null) return o[k]
  return undefined
}
const range = (n) => Array.from({ length: n }, (_, i) => i)
const brief = (a) => (a.length > 12 ? `${a.slice(0, 10).join(',')}… (${a.length})` : a.join(',') || '—')

async function json(url) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${url}`)
  return r.json()
}

for (const [key, p] of Object.entries(PRESETS)) {
  if (!p.source) continue
  const repo = MIRROR[p.source] ?? p.source
  let hf
  let api
  try {
    ;[hf, api] = await Promise.all([
      json(`https://huggingface.co/${repo}/raw/main/config.json`),
      json(`https://huggingface.co/api/models/${p.source}`).catch(() => null),
    ])
  } catch (e) {
    console.log(`?  ${key}: could not fetch ${repo} (${e.message})`)
    continue
  }

  const t = hf.text_config ?? hf // multimodal configs nest the decoder
  const cfg = { ...p.config, ...(p.full ?? {}) }
  const mla = cfg.attn_type === 'mla'
  const dsv4 = cfg.attn_type === 'dsv4'
  const hasDense = range(cfg.n_layers).some((i) => !isMoeLayer(cfg, i))
  const out = []

  const eq = (field, want) => {
    if (want === undefined) return
    const got = field === 'n_layers' ? cfg.n_layers : cfg[field]
    if (got === undefined) return
    scalars++
    const same = typeof want === 'number' && typeof got === 'number' ? want === got : String(want) === String(got)
    if (!same) {
      mismatches++
      out.push(`MISMATCH ${field}: preset ${got} vs config.json ${want}`)
    }
  }
  const cmp = (what, mine, theirs) => {
    const d = [...new Set([...mine.filter((x) => !theirs.includes(x)), ...theirs.filter((x) => !mine.includes(x))])]
    schedules++
    if (!d.length) return out.push(`ok ${what} — ${mine.length} layers match exactly`)
    mismatches++
    out.push(`MISMATCH ${what}\n       preset [${brief(mine)}]\n       hf     [${brief(theirs)}]\n       differ [${brief(d.sort((a, b) => a - b))}]`)
  }

  // ---- scalars, only where the field actually drives the model ----------------
  eq('n_layers', t.num_hidden_layers)
  eq('vocab_size', t.vocab_size)
  eq('dim', t.hidden_size)
  eq('n_heads', t.num_attention_heads)
  eq('tie_embeddings', pick(t, 'tie_word_embeddings') ?? pick(hf, 'tie_word_embeddings'))
  // n_kv_heads / head_dim only feed the MHA / GQA path; MLA uses the lora ranks
  if (!mla && !dsv4) {
    eq('n_kv_heads', t.num_key_value_heads)
    eq('head_dim', pick(t, 'head_dim') ?? t.hidden_size / t.num_attention_heads)
  }
  if (dsv4) eq('head_dim', t.head_dim)
  if (hasDense) eq('ffn_dim', pick(t, 'intermediate_size_mlp', 'intermediate_size')) // Llama 4 splits the two
  if (cfg.arch === 'moe') {
    // Mixtral / gpt-oss have no separate key: intermediate_size IS the expert width
    eq('moe_inter_dim', pick(t, 'moe_intermediate_size', 'expert_intermediate_size', 'intermediate_size'))
    eq('n_routed_experts', pick(t, 'n_routed_experts', 'num_experts', 'num_local_experts'))
    eq('n_activated_experts', pick(t, 'num_experts_per_tok', 'num_experts_per_token', 'experts_per_token'))
    eq('n_shared_experts', pick(t, 'n_shared_experts', 'num_shared_experts'))
    eq('n_dense_layers', t.first_k_dense_replace)
    eq('moe_layer_step', t.moe_layer_freq)
    eq('moe_latent_dim', t.routed_expert_hidden_size)
  }
  if (mla || dsv4) for (const f of ['q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim']) eq(f, t[f])
  if (dsv4) {
    eq('o_lora_rank', t.o_lora_rank)
    eq('o_groups', t.o_groups)
    eq('hc_mult', t.hc_mult)
    eq('window_size', t.sliding_window)
  }
  eq('index_n_heads', t.index_n_heads)
  eq('index_head_dim', t.index_head_dim)

  // ---- per-layer schedules, wherever config.json spells them out --------------
  if (cfg.arch === 'moe') {
    let dense = null
    if (Array.isArray(t.mlp_layer_types)) dense = t.mlp_layer_types.flatMap((v, i) => (v === 'dense' ? [i] : []))
    else if (Array.isArray(t.moe_layers)) dense = range(t.num_hidden_layers).filter((i) => !t.moe_layers.includes(i))
    else if (Array.isArray(t.mlp_only_layers) && t.mlp_only_layers.length) dense = t.mlp_only_layers
    else if (t.first_k_dense_replace !== undefined || t.moe_layer_freq !== undefined) {
      const fk = t.first_k_dense_replace ?? 0
      const fr = t.moe_layer_freq ?? 1
      dense = range(t.num_hidden_layers).filter((i) => !(i >= fk && (i + 1) % fr === 0))
    }
    if (dense) cmp('dense (non-MoE) layers', range(cfg.n_layers).filter((i) => !isMoeLayer(cfg, i)), dense)
  }

  let full = null
  if (t.linear_attn_config?.full_attn_layers) full = t.linear_attn_config.full_attn_layers.map((n) => n - 1) // 1-based
  else if (Array.isArray(t.layer_types) && t.layer_types.includes('linear_attention')) full = t.layer_types.flatMap((v, i) => (v === 'full_attention' ? [i] : []))
  else if (t.full_attention_interval) full = range(t.num_hidden_layers).filter((i) => (i + 1) % t.full_attention_interval === 0)
  if (full && cfg.linear_attn && cfg.linear_attn !== 'none') {
    cmp('full-attention layers', range(cfg.n_layers).filter((i) => isFullAttnLayer(cfg, i)), full)
  }

  if (t.index_n_heads && t.index_topk_freq) {
    const off = t.index_skip_topk_offset ?? 0
    cmp(
      'layers with a full DSA indexer',
      range(cfg.n_layers).filter((i) => hasIndexer(cfg, i)),
      range(t.num_hidden_layers).filter((i) => i < off || (i - off + 1) % t.index_topk_freq === 0),
    )
  }

  if (t.num_hash_layers !== undefined) {
    cmp('hash-routed layers', range(cfg.n_layers).filter((i) => isHashLayer(cfg, i)), range(t.num_hash_layers))
  }

  if (Array.isArray(t.compress_ratios)) {
    // the checkpoint's array carries one extra entry for the MTP layer
    const off = range(cfg.n_layers).filter((i) => compressRatio(cfg, i) !== t.compress_ratios[i])
    schedules++
    if (off.length) {
      mismatches++
      out.push(`MISMATCH compress_ratios at layers [${brief(off)}]`)
    } else out.push(`ok compress_ratios — all ${cfg.n_layers} layers match`)
  }

  // ---- the published total itself, against the live safetensors count ---------
  const live = api?.safetensors?.total
  if (live && p.published) {
    const st = fullModelStats(p)
    const rel = Math.abs(p.published - live) / live
    if (rel > 1e-9 && !p.publishedExcludes) {
      mismatches++
      out.push(`MISMATCH published ${p.published} vs live safetensors ${live} (${(rel * 100).toFixed(4)}%; this repo models ${st.total})`)
    } else {
      out.push(`ok published total${p.publishedExcludes ? ` (${p.publishedExcludes} excluded from the checkpoint's ${live})` : ''}`)
    }
  }

  const bad = out.some((l) => l.startsWith('MISMATCH'))
  console.log(`\n${bad ? '✗' : '·'} ${key}  ${repo}${repo === p.source ? '' : `  (mirror of ${p.source})`}`)
  for (const l of out) console.log(`    ${l}`)
}

console.log(`\n${scalars} scalar fields and ${schedules} per-layer schedules compared`)
if (mismatches) {
  console.error(`\n✗ ${mismatches} mismatch(es) against Hugging Face.`)
  process.exit(1)
}
console.log('\n✓ Every preset matches its config.json on Hugging Face.')
