// Model presets. Every preset is truncated to a small number of layers so the
// FQN table stays readable; all per-layer shapes match the reference model.

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

  tie_embeddings: true,
}

export const PRESETS = {
  'tiny-dense': {
    label: 'Tiny dense (playground)',
    config: { ...BASE, arch: 'dense' },
  },
  'tiny-moe': {
    label: 'Tiny MoE (playground)',
    config: { ...BASE, arch: 'moe' },
  },
  'llama3-8b': {
    label: 'Llama-3 8B shapes',
    config: {
      ...BASE,
      arch: 'dense',
      vocab_size: 128256,
      dim: 4096,
      attn_type: 'gqa',
      n_heads: 32,
      n_kv_heads: 8,
      head_dim: 128,
      ffn_dim: 14336,
      tie_embeddings: false,
    },
  },
  'qwen3-30b-a3b': {
    label: 'Qwen3 30B-A3B shapes',
    config: {
      ...BASE,
      arch: 'moe',
      vocab_size: 151936,
      dim: 2048,
      attn_type: 'gqa',
      n_heads: 32,
      n_kv_heads: 4,
      head_dim: 128,
      qk_norm: true,
      moe_inter_dim: 768,
      n_routed_experts: 128,
      n_shared_experts: 0,
      n_activated_experts: 8,
      balance_bias: false,
      tie_embeddings: false,
    },
  },
  'gpt-oss-20b': {
    label: 'GPT-OSS 20B shapes',
    config: {
      ...BASE,
      arch: 'moe',
      vocab_size: 201088,
      dim: 2880,
      attn_type: 'gqa',
      n_heads: 64,
      n_kv_heads: 8,
      head_dim: 64,
      qkv_bias: true,
      o_bias: true,
      attn_sinks: true,
      moe_inter_dim: 2880,
      n_routed_experts: 32,
      n_shared_experts: 0,
      n_activated_experts: 4,
      router_bias: true,
      expert_bias: true,
      balance_bias: false,
      tie_embeddings: false,
    },
  },
  'deepseek-v3': {
    label: 'DeepSeek-V3 shapes',
    config: {
      ...BASE,
      arch: 'moe',
      vocab_size: 129280,
      dim: 7168,
      attn_type: 'mla',
      n_heads: 128,
      q_lora_rank: 1536,
      kv_lora_rank: 512,
      qk_nope_head_dim: 128,
      qk_rope_head_dim: 64,
      v_head_dim: 128,
      ffn_dim: 18432,
      moe_inter_dim: 2048,
      n_routed_experts: 256,
      n_shared_experts: 1,
      n_activated_experts: 8,
      n_dense_layers: 1,
      tie_embeddings: false,
    },
  },
}

export const DEFAULT_PRESET = 'tiny-moe'
