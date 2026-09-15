// Model presets. Each one is truncated to 2 layers so the FQN table stays
// readable (one dense + one MoE layer where the model has both); per-layer
// shapes come from the model's config.json and safetensors headers on Hugging
// Face. `full` restores the real depth and `published` is the Hugging Face
// safetensors parameter total, checked by `npm run validate`.

import { buildModel, isMoeLayer } from './model.js'
import { prod } from './format.js'

const BASE = {
  arch: 'dense', // 'dense' | 'moe'
  vocab_size: 2048,
  dim: 256,
  n_layers: 2,

  // Attention
  attn_type: 'gqa', // 'mha' | 'gqa' | 'mla'
  n_heads: 8,
  n_kv_heads: 2,
  head_dim: 32,
  qkv_bias: false,
  o_bias: false,
  qk_norm: false,
  attn_sinks: false,

  // MLA (DeepSeek V3). q_lora_rank = 0 means a direct (non low-rank) query proj.
  q_lora_rank: 128,
  kv_lora_rank: 64,
  qk_nope_head_dim: 16,
  qk_rope_head_dim: 16,
  v_head_dim: 32,

  // Dense FFN (SwiGLU)
  ffn_dim: 1024,
  mlp_bias: false,

  // MoE (SwiGLU experts)
  moe_inter_dim: 256,
  n_routed_experts: 8,
  n_shared_experts: 1,
  n_activated_experts: 2,
  n_dense_layers: 0, // first-k layers use a dense FFN (DeepSeek V3 style)
  expert_layout: 'grouped', // 'grouped' ([E, ...] tensors) | 'module_list' (experts.{e}.*)
  router_bias: false, // learned router bias parameter (GPT-OSS)
  expert_bias: false, // per-expert linear biases (GPT-OSS)
  balance_bias: true, // aux-loss-free balancing buffer (DeepSeek V3)

  moe_layer_step: 1, // every n-th layer is MoE (Llama 4 Maverick: 2)
  qk_norm_type: 'head', // 'head' ([head_dim]) | 'full' ([heads × head_dim], MiniMax-M2)

  // DSA lightning indexer (DeepSeek-V3.2, GLM-5.x); 0 heads = off
  index_n_heads: 0,
  index_head_dim: 128,
  index_topk: 2048,
  index_full_first: 0,
  index_freq: 1,

  // Hybrid linear attention (Kimi K3 KDA, Qwen3-Next / Qwen3.8 gated DeltaNet)
  linear_attn: 'none', // 'none' | 'kda' | 'gdn'
  full_attn_period: 4, // every n-th layer (1-based) keeps full attention
  full_attn_last: false, // the last layer is always full attention (Kimi K3)
  la_num_heads: 16, // KDA heads / gated DeltaNet value heads
  la_num_k_heads: 8, // gated DeltaNet key heads
  la_head_dim: 32, // KDA head dim / gated DeltaNet key head dim
  la_v_head_dim: 32, // gated DeltaNet value head dim
  la_conv: 4, // short convolution kernel
  attn_output_gate: false, // sigmoid gate on the attention output (Qwen3-Next, Kimi K3)
  attn_res: false, // attention residuals over earlier layers (Kimi K3)
  moe_latent_dim: 0, // routed experts run in a latent width (Kimi K3); 0 = off
  shared_expert_gate: false, // sigmoid gate on the shared expert (Qwen3-Next)

  // DeepSeek-V4 attention (attn_type 'dsv4'); reuses n_heads, head_dim,
  // q_lora_rank, qk_rope_head_dim and the index_* fields
  o_lora_rank: 64,
  o_groups: 4,
  window_size: 128,
  compress_ratios: [0, 4, 128], // per layer: 0 window only, 4 compressed + indexer, 128 compressed
  hc_mult: 4, // hyper-connection residual copies
  n_hash_layers: 0, // leading layers routed by token-id lookup

  tie_embeddings: true,
}

const dense = (c) => ({ ...BASE, arch: 'dense', ...c })
const moe = (c) => ({ ...BASE, arch: 'moe', ...c })
const gqa = { attn_type: 'gqa', qk_norm: false, qkv_bias: false, o_bias: false, attn_sinks: false }
const mla = (c) => ({ attn_type: 'mla', ...c })
const dsv4 = (c) => ({ attn_type: 'dsv4', ...c })
// DeepSeek-style MoE: sigmoid router + e_score_correction_bias buffer, shared expert
const dsMoe = { balance_bias: true, router_bias: false, expert_bias: false }
const plainMoe = { balance_bias: false, router_bias: false, expert_bias: false, n_shared_experts: 0 }

export const PRESET_GROUPS = ['Playground', 'DeepSeek', 'Moonshot', 'Qwen', 'Z.ai', 'MiniMax', 'OpenAI', 'Meta', 'Mistral']

// DeepSeek-V4 per-layer KV compression, including the MTP layer at the end.
const V4_FLASH_RATIOS = [0, 0, ...Array.from({ length: 40 }, (_, i) => (i % 2 ? 128 : 4)), 4, 0]
const V4_PRO_RATIOS = [128, 128, ...Array.from({ length: 58 }, (_, i) => (i % 2 ? 128 : 4)), 4, 0]

export const PRESETS = {
  'tiny-dense': {
    label: 'Tiny dense',
    group: 'Playground',
    config: dense({}),
  },
  'tiny-moe': {
    label: 'Tiny MoE',
    group: 'Playground',
    config: moe({}),
  },

  // ---- Meta ------------------------------------------------------------------
  'llama3-8b': {
    label: 'Llama 3.1 8B',
    group: 'Meta',
    source: 'meta-llama/Llama-3.1-8B',
    config: dense({ ...gqa, vocab_size: 128256, dim: 4096, n_heads: 32, n_kv_heads: 8, head_dim: 128, ffn_dim: 14336, tie_embeddings: false }),
    full: { n_layers: 32 },
    published: 8_030_261_248,
  },
  'llama3.3-70b': {
    label: 'Llama 3.3 70B',
    group: 'Meta',
    source: 'meta-llama/Llama-3.3-70B-Instruct',
    config: dense({ ...gqa, vocab_size: 128256, dim: 8192, n_heads: 64, n_kv_heads: 8, head_dim: 128, ffn_dim: 28672, tie_embeddings: false }),
    full: { n_layers: 80 },
    published: 70_553_706_496,
  },
  'llama4-scout': {
    label: 'Llama 4 Scout 17B-16E',
    group: 'Meta',
    source: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    note: 'text model only; the published total includes the vision encoder',
    config: moe({ ...gqa, ...plainMoe, vocab_size: 202048, dim: 5120, n_heads: 40, n_kv_heads: 8, head_dim: 128, ffn_dim: 16384, moe_inter_dim: 8192, n_routed_experts: 16, n_activated_experts: 1, n_shared_experts: 1, n_dense_layers: 0, moe_layer_step: 1, tie_embeddings: false }),
    full: { n_layers: 48 },
    published: 108_641_793_536,
    publishedExtra: 'vision',
  },
  'llama4-maverick': {
    label: 'Llama 4 Maverick 17B-128E',
    group: 'Meta',
    source: 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    note: 'dense and MoE layers alternate; the published total includes the vision encoder',
    config: moe({ ...gqa, ...plainMoe, vocab_size: 202048, dim: 5120, n_heads: 40, n_kv_heads: 8, head_dim: 128, ffn_dim: 16384, moe_inter_dim: 8192, n_routed_experts: 128, n_activated_experts: 1, n_shared_experts: 1, n_dense_layers: 0, moe_layer_step: 2, tie_embeddings: false }),
    full: { n_layers: 48 },
    published: 401_583_781_376,
    publishedExtra: 'vision',
  },

  // ---- Qwen ------------------------------------------------------------------
  'qwen3-32b': {
    label: 'Qwen3 32B',
    group: 'Qwen',
    source: 'Qwen/Qwen3-32B',
    config: dense({ ...gqa, qk_norm: true, vocab_size: 151936, dim: 5120, n_heads: 64, n_kv_heads: 8, head_dim: 128, ffn_dim: 25600, tie_embeddings: false }),
    full: { n_layers: 64 },
    published: 32_762_123_264,
  },
  'qwen3-30b-a3b': {
    label: 'Qwen3 30B-A3B',
    group: 'Qwen',
    source: 'Qwen/Qwen3-30B-A3B',
    config: moe({ ...gqa, ...plainMoe, qk_norm: true, vocab_size: 151936, dim: 2048, n_heads: 32, n_kv_heads: 4, head_dim: 128, ffn_dim: 6144, moe_inter_dim: 768, n_routed_experts: 128, n_activated_experts: 8, tie_embeddings: false }),
    full: { n_layers: 48 },
    published: 30_532_122_624,
  },
  'qwen3-235b-a22b': {
    label: 'Qwen3 235B-A22B',
    group: 'Qwen',
    source: 'Qwen/Qwen3-235B-A22B',
    config: moe({ ...gqa, ...plainMoe, qk_norm: true, vocab_size: 151936, dim: 4096, n_heads: 64, n_kv_heads: 4, head_dim: 128, ffn_dim: 12288, moe_inter_dim: 1536, n_routed_experts: 128, n_activated_experts: 8, tie_embeddings: false }),
    full: { n_layers: 94 },
    published: 235_093_634_560,
  },
  'qwen3.8-2.4t': {
    label: 'Qwen3.8 2.4T-A95B',
    group: 'Qwen',
    source: 'Qwen/Qwen3.8-2.4T-A95B',
    note: 'real model has full attention every 4th layer; shown as gated DeltaNet then gated attention',
    config: moe({
      ...gqa, ...plainMoe, qk_norm: true, attn_output_gate: true,
      linear_attn: 'gdn', full_attn_period: 2, la_num_heads: 128, la_num_k_heads: 16, la_head_dim: 128, la_v_head_dim: 128, la_conv: 4,
      n_shared_experts: 1, shared_expert_gate: true,
      vocab_size: 248320, dim: 8192, n_heads: 64, n_kv_heads: 4, head_dim: 256, moe_inter_dim: 2048, n_routed_experts: 512, n_activated_experts: 10, tie_embeddings: false,
    }),
    full: { n_layers: 92, full_attn_period: 4, mtp_layers: 1, mtp_full_attn: true },
    published: 2_446_182_725_504,
  },
  'qwen3-next-80b': {
    label: 'Qwen3-Next 80B-A3B',
    group: 'Qwen',
    source: 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    note: 'real model has full attention every 4th layer; shown as gated DeltaNet then gated attention',
    config: moe({
      ...gqa, ...plainMoe, qk_norm: true, attn_output_gate: true,
      linear_attn: 'gdn', full_attn_period: 2, la_num_heads: 32, la_num_k_heads: 16, la_head_dim: 128, la_v_head_dim: 128, la_conv: 4,
      n_shared_experts: 1, shared_expert_gate: true,
      vocab_size: 151936, dim: 2048, n_heads: 16, n_kv_heads: 2, head_dim: 256, moe_inter_dim: 512, n_routed_experts: 512, n_activated_experts: 10, tie_embeddings: false,
    }),
    full: { n_layers: 48, full_attn_period: 4, mtp_layers: 1, mtp_full_attn: true },
    published: 81_324_862_720,
  },
  'qwen3-coder-480b': {
    label: 'Qwen3-Coder 480B-A35B',
    group: 'Qwen',
    source: 'Qwen/Qwen3-Coder-480B-A35B-Instruct',
    config: moe({ ...gqa, ...plainMoe, qk_norm: true, vocab_size: 151936, dim: 6144, n_heads: 96, n_kv_heads: 8, head_dim: 128, ffn_dim: 8192, moe_inter_dim: 2560, n_routed_experts: 160, n_activated_experts: 8, tie_embeddings: false }),
    full: { n_layers: 62 },
    published: 480_154_875_392,
  },

  // ---- DeepSeek ----------------------------------------------------------------
  'deepseek-v3': {
    label: 'DeepSeek-V3 / V3.1',
    group: 'DeepSeek',
    source: 'deepseek-ai/DeepSeek-V3',
    config: moe({
      ...dsMoe,
      ...mla({ n_heads: 128, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 }),
      vocab_size: 129280, dim: 7168, ffn_dim: 18432, moe_inter_dim: 2048, n_routed_experts: 256, n_activated_experts: 8, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false,
    }),
    full: { n_layers: 61, n_dense_layers: 3, mtp_layers: 1, mtp_embed_copies: true },
    published: 684_531_386_000,
  },
  'deepseek-v3.2': {
    label: 'DeepSeek-V3.2 (DSA)',
    group: 'DeepSeek',
    source: 'deepseek-ai/DeepSeek-V3.2',
    config: moe({
      ...dsMoe,
      ...mla({ n_heads: 128, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 }),
      index_n_heads: 64, index_head_dim: 128, index_topk: 2048, index_full_first: 0, index_freq: 1,
      vocab_size: 129280, dim: 7168, ffn_dim: 18432, moe_inter_dim: 2048, n_routed_experts: 256, n_activated_experts: 8, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false,
    }),
    full: { n_layers: 61, n_dense_layers: 3, mtp_layers: 1, mtp_embed_copies: true },
    published: 685_396_921_376,
  },

  'deepseek-v4-flash': {
    label: 'DeepSeek-V4-Flash 284B',
    group: 'DeepSeek',
    source: 'deepseek-ai/DeepSeek-V4-Flash',
    note: '3 layers shown: sliding window with hash routing, ×4 compressed KV with indexer, ×128 compressed KV',
    config: moe({
      ...dsMoe,
      ...dsv4({ n_heads: 64, head_dim: 512, qk_rope_head_dim: 64, q_lora_rank: 1024, o_lora_rank: 1024, o_groups: 8, window_size: 128, index_n_heads: 64, index_head_dim: 128, index_topk: 512, hc_mult: 4 }),
      compress_ratios: [0, 4, 128], n_hash_layers: 1, n_layers: 3,
      vocab_size: 129280, dim: 4096, moe_inter_dim: 2048, n_routed_experts: 256, n_activated_experts: 6, n_shared_experts: 1, n_dense_layers: 0, tie_embeddings: false,
    }),
    full: { n_layers: 43, n_hash_layers: 3, compress_ratios: V4_FLASH_RATIOS, mtp_layers: 1 },
    published: 290_944_616_402,
  },
  'deepseek-v4-pro': {
    label: 'DeepSeek-V4-Pro 1.6T',
    group: 'DeepSeek',
    source: 'deepseek-ai/DeepSeek-V4-Pro',
    note: '3 layers shown: ×128 compressed KV with hash routing, ×4 compressed KV with indexer, ×128 compressed KV',
    config: moe({
      ...dsMoe,
      ...dsv4({ n_heads: 128, head_dim: 512, qk_rope_head_dim: 64, q_lora_rank: 1536, o_lora_rank: 1024, o_groups: 16, window_size: 128, index_n_heads: 64, index_head_dim: 128, index_topk: 1024, hc_mult: 4 }),
      compress_ratios: [128, 4, 128], n_hash_layers: 1, n_layers: 3,
      vocab_size: 129280, dim: 7168, moe_inter_dim: 3072, n_routed_experts: 384, n_activated_experts: 6, n_shared_experts: 1, n_dense_layers: 0, tie_embeddings: false,
    }),
    full: { n_layers: 61, n_hash_layers: 3, compress_ratios: V4_PRO_RATIOS, mtp_layers: 1 },
    published: 1_598_839_674_782,
  },

  // ---- Moonshot ----------------------------------------------------------------
  'kimi-k3': {
    label: 'Kimi K3 2.8T',
    group: 'Moonshot',
    source: 'moonshotai/Kimi-K3',
    note: '3 layers shown: KDA + dense MLP, KDA + latent MoE, gated MLA + latent MoE; text weights only (vision encoder excluded)',
    config: moe({
      ...dsMoe,
      ...mla({ n_heads: 96, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 }),
      attn_output_gate: true, attn_res: true,
      linear_attn: 'kda', full_attn_period: 4, full_attn_last: true, la_num_heads: 96, la_head_dim: 128, la_conv: 4,
      moe_latent_dim: 3584, n_layers: 3,
      vocab_size: 163840, dim: 7168, ffn_dim: 33792, moe_inter_dim: 3072, n_routed_experts: 896, n_activated_experts: 16, n_shared_experts: 2, n_dense_layers: 1, tie_embeddings: false,
    }),
    full: { n_layers: 93 },
    published: 2_779_484_478_208,
  },
  'kimi-k2': {
    label: 'Kimi K2 / K2.6 1T-A32B',
    group: 'Moonshot',
    source: 'moonshotai/Kimi-K2-Instruct',
    config: moe({
      ...dsMoe,
      ...mla({ n_heads: 64, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 }),
      vocab_size: 163840, dim: 7168, ffn_dim: 18432, moe_inter_dim: 2048, n_routed_experts: 384, n_activated_experts: 8, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false,
    }),
    full: { n_layers: 61 },
    published: 1_026_408_235_864,
  },

  // ---- Z.ai ----------------------------------------------------------------------
  'glm-4.5-air': {
    label: 'GLM-4.5-Air 106B-A12B',
    group: 'Z.ai',
    source: 'zai-org/GLM-4.5-Air',
    config: moe({ ...gqa, ...dsMoe, qkv_bias: true, vocab_size: 151552, dim: 4096, n_heads: 96, n_kv_heads: 8, head_dim: 128, ffn_dim: 10944, moe_inter_dim: 1408, n_routed_experts: 128, n_activated_experts: 8, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false }),
    full: { n_layers: 46, mtp_layers: 1, mtp_embed_copies: true },
    published: 110_468_824_832,
  },
  'glm-4.5': {
    label: 'GLM-4.5 355B-A32B',
    group: 'Z.ai',
    source: 'zai-org/GLM-4.5',
    config: moe({ ...gqa, ...dsMoe, qkv_bias: true, qk_norm: true, vocab_size: 151552, dim: 5120, n_heads: 96, n_kv_heads: 8, head_dim: 128, ffn_dim: 12288, moe_inter_dim: 1536, n_routed_experts: 160, n_activated_experts: 8, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false }),
    full: { n_layers: 92, n_dense_layers: 3, mtp_layers: 1, mtp_embed_copies: true },
    published: 358_337_791_296,
  },
  'glm-4.7-flash': {
    label: 'GLM-4.7-Flash 30B-A3B',
    group: 'Z.ai',
    source: 'zai-org/GLM-4.7-Flash',
    config: moe({
      ...dsMoe,
      ...mla({ n_heads: 20, q_lora_rank: 768, kv_lora_rank: 512, qk_nope_head_dim: 192, qk_rope_head_dim: 64, v_head_dim: 256 }),
      vocab_size: 154880, dim: 2048, ffn_dim: 10240, moe_inter_dim: 1536, n_routed_experts: 64, n_activated_experts: 4, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false,
    }),
    full: { n_layers: 47, mtp_layers: 1, mtp_embed_copies: true },
    published: 31_221_488_576,
  },
  'glm-5.3': {
    label: 'GLM-5.3 (DSA)',
    group: 'Z.ai',
    source: 'zai-org/GLM-5.3',
    config: moe({
      ...dsMoe,
      ...mla({ n_heads: 64, q_lora_rank: 2048, kv_lora_rank: 512, qk_nope_head_dim: 192, qk_rope_head_dim: 64, v_head_dim: 256 }),
      index_n_heads: 32, index_head_dim: 128, index_topk: 2048, index_full_first: 3, index_freq: 4,
      vocab_size: 154880, dim: 6144, ffn_dim: 12288, moe_inter_dim: 2048, n_routed_experts: 256, n_activated_experts: 8, n_shared_experts: 1, n_dense_layers: 1, tie_embeddings: false,
    }),
    full: { n_layers: 78, n_dense_layers: 3, mtp_layers: 1 },
    published: 753_329_940_480,
  },

  // ---- OpenAI --------------------------------------------------------------------
  'gpt-oss-20b': {
    label: 'gpt-oss-20b',
    group: 'OpenAI',
    source: 'openai/gpt-oss-20b',
    config: moe({ ...gqa, qkv_bias: true, o_bias: true, attn_sinks: true, balance_bias: false, router_bias: true, expert_bias: true, n_shared_experts: 0, vocab_size: 201088, dim: 2880, n_heads: 64, n_kv_heads: 8, head_dim: 64, moe_inter_dim: 2880, n_routed_experts: 32, n_activated_experts: 4, tie_embeddings: false }),
    full: { n_layers: 24 },
    published: 20_914_757_184,
    publishedExtra: 'mxfp4',
  },
  'gpt-oss-120b': {
    label: 'gpt-oss-120b',
    group: 'OpenAI',
    source: 'openai/gpt-oss-120b',
    config: moe({ ...gqa, qkv_bias: true, o_bias: true, attn_sinks: true, balance_bias: false, router_bias: true, expert_bias: true, n_shared_experts: 0, vocab_size: 201088, dim: 2880, n_heads: 64, n_kv_heads: 8, head_dim: 64, moe_inter_dim: 2880, n_routed_experts: 128, n_activated_experts: 4, tie_embeddings: false }),
    full: { n_layers: 36 },
    published: 116_829_156_672,
    publishedExtra: 'mxfp4',
  },

  // ---- MiniMax -------------------------------------------------------------------
  'minimax-m2': {
    label: 'MiniMax-M2 / M2.7 230B-A10B',
    group: 'MiniMax',
    source: 'MiniMaxAI/MiniMax-M2',
    config: moe({ ...gqa, ...dsMoe, qk_norm: true, qk_norm_type: 'full', n_shared_experts: 0, vocab_size: 200064, dim: 3072, n_heads: 48, n_kv_heads: 8, head_dim: 128, moe_inter_dim: 1536, n_routed_experts: 256, n_activated_experts: 8, tie_embeddings: false }),
    full: { n_layers: 62 },
    published: 228_689_764_864,
  },

  // ---- Mistral -------------------------------------------------------------------
  'mixtral-8x22b': {
    label: 'Mixtral 8x22B',
    group: 'Mistral',
    source: 'mistralai/Mixtral-8x22B-v0.1',
    config: moe({ ...gqa, ...plainMoe, vocab_size: 32000, dim: 6144, n_heads: 48, n_kv_heads: 8, head_dim: 128, moe_inter_dim: 16384, n_routed_experts: 8, n_activated_experts: 2, tie_embeddings: false }),
    full: { n_layers: 56 },
    published: 140_620_634_112,
  },
}

export const DEFAULT_PRESET = 'tiny-moe'

// Whole-model counts at the real depth (`full` overrides any config field).
// Buffers stored in checkpoints (router balance bias, hash tables) count too.
// An MTP module is one extra decoder layer plus its input projections [d, 2d]
// and three norms; DeepSeek-V3 / GLM-4.x MTP layers also store embedding and
// LM-head copies (`mtp_embed_copies`), DeepSeek-V4 adds a hyper-connection head.
export function fullModelStats(preset) {
  if (!preset.full) return null
  const { mtp_layers: mtp = 0, mtp_embed_copies: copies = false, mtp_full_attn, ...over } = preset.full
  const cfg = { ...preset.config, ...over }
  const params = buildModel(cfg)
  const sum = (ps) => ps.reduce((a, p) => a + prod(p.shape), 0)
  const D = cfg.dim
  let mtpLayer = 0
  if (mtp) {
    const n = cfg.n_layers
    const ext = buildModel({ ...cfg, n_layers: n + 1, mtp_layer_index: n, mtp_full_attn })
    mtpLayer = sum(ext.filter((p) => p.layer === n)) + 2 * D * D + 3 * D
    if (copies) mtpLayer += 2 * cfg.vocab_size * D
    if (cfg.attn_type === 'dsv4') mtpLayer += cfg.hc_mult * cfg.hc_mult * D + cfg.hc_mult + 1
  }
  const total = sum(params) + mtp * mtpLayer
  let active = total
  if (cfg.arch === 'moe') {
    const experts = sum(params.filter((p) => p.group === 'experts'))
    active = total - mtp * mtpLayer - experts + (experts * cfg.n_activated_experts) / cfg.n_routed_experts
  }
  const moeLayers = Array.from({ length: cfg.n_layers }, (_, i) => isMoeLayer(cfg, i)).filter(Boolean).length
  return { total, active, layers: cfg.n_layers, moeLayers, mtp }
}
