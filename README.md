# Shardlock

*Where did my tensors go?*

An interactive visualizer for how a small (2-layer by default) **dense** or **sparse (MoE)**
transformer is laid out across GPUs under **DDP, FSDP, HSDP, TP, EP and CP**. For every FQN it shows
the global shape, the local shape on any chosen rank, the placement chain (e.g.
`EP Shard(0)/8 → TP Shard(1)/8 → FSDP Shard(0)/2`), parameter counts and bytes.

Everything runs client-side (React 19 + Vite + zustand) and deploys to GitHub Pages from `docs/`.

## Views

| View | What it shows |
| --- | --- |
| **Device mesh** | Every rank as a cell, organized `[dp, cp, tp]`. Click a rank to inspect it; lenses highlight its TP / CP / DP / EP / expert-DP / FSDP-shard group. |
| **Parameters** | Module tree of FQNs with global vs local shapes, placement, counts and bytes, plus a shard diagram for the selected tensor (EP / TP / FSDP boundaries, this rank's shard highlighted, hover for the ranks holding a shard). |
| **Layer flow** | One transformer block as a data-flow graph on the selected rank: every module with its local params (global → local shape, bytes), collectives inline where they fire (SP/TP region boundaries, CP ring / all-gather KV, EP dispatch / combine, FSDP/HSDP/DDP param + grad collectives), and edges labeled with local tensor shapes (numbers or symbols like `[b, s/(cp·tp), d]`). Toggle **Forward / Backward**: backward reverses the arrows, shows ∂ shapes and ∂W sizes, and swaps each collective for its conjugate. Click any node for placement, group ranks and what is communicated. |
| **Memory** | Per-rank weights / grads / optimizer / master weights / FSDP all-gather transient, a DDP vs FSDP vs HSDP comparison, and FSDP unit sizes. |
| **Activations** | Forward tensor shapes on the rank for one micro-batch: where CP shards the sequence, SP shards norm regions, TP shards heads/hidden, EP dispatches tokens. |
| **Communication** | Collectives per step with group, buffer size, call count and volume (TP / SP, ring or all-gather CP, EP all-to-all, FSDP all-gather / reduce-scatter, HSDP cross-block all-reduce, DDP buckets). |

## Model

FQNs follow the `moe_model` reference module (`~/dev/git/parallelism-visualizer`):

```
tok_embeddings.weight
layers.{i}.attn_norm.weight
layers.{i}.attn.{wq,wk,wv,wo}.weight              # MHA / GQA (+ optional biases, qk-norm, sinks)
layers.{i}.attn.{q_a_proj,q_a_norm,q_b_proj,kv_a_proj,kv_a_norm,kv_b_proj,wo}   # MLA
layers.{i}.mlp_norm.weight, layers.{i}.mlp.{w_gate,w_up,w_down}.weight          # dense
layers.{i}.moe_norm.weight, layers.{i}.moe.router.{weight,balance_bias}         # MoE
layers.{i}.moe.experts.{w_gate,w_up,w_down}        # grouped [E, …] tensors, or
layers.{i}.moe.experts.{e}.{w_gate,w_up,w_down}.weight   # ModuleList layout
layers.{i}.moe.shared_experts.{w_gate,w_up,w_down}.weight
final_norm.weight
lm_head.weight                                     # omitted when tied
```

### Presets

Every preset is taken from the model's `config.json`, safetensors headers and (where the architecture is new)
its reference modeling code on Hugging Face, and shown truncated to 2–3 layers so each distinct layer type
appears once. `npm run validate` rebuilds each preset at its real depth and checks the parameter total against
the published safetensors count.

| Group | Presets |
| --- | --- |
| Playground | Tiny dense, Tiny MoE (matches `MoEDecoder`, 4,396,288 params) |
| DeepSeek | DeepSeek-V4-Flash, DeepSeek-V4-Pro, DeepSeek-V3 / V3.1, DeepSeek-V3.2 (MLA + DSA indexer) |
| Moonshot | Kimi K3 (KDA / gated-MLA hybrid, latent MoE, attention residuals), Kimi K2 / K2.6 |
| Qwen | Qwen3.8 2.4T-A95B and Qwen3-Next 80B-A3B (gated DeltaNet hybrid), Qwen3 32B / 30B-A3B / 235B-A22B, Qwen3-Coder 480B |
| Z.ai | GLM-5.3 (MLA + DSA indexer), GLM-4.7-Flash, GLM-4.5, GLM-4.5-Air |
| MiniMax | MiniMax-M2 / M2.7 (q/k norm over all heads) |
| OpenAI | gpt-oss-20b, gpt-oss-120b |
| Meta | Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout, Llama 4 Maverick (interleaved MoE) |
| Mistral | Mixtral 8x22B |

Totals match exactly for the new architectures and to within 0.01% elsewhere (DeepSeek-V3's published count
includes FP8 scale tensors). Llama 4 is ~0.87B lower because its checkpoint also contains the vision encoder; Kimi K3
is compared against its text weights. MTP modules are counted.

**Block types**

- **DeepSeek-V4** — low-rank q, single-head window KV, grouped low-rank output (`wo_a` / `wo_b`), per-layer KV
  compressor (×4 with a lightning indexer, ×128 without), attention sinks, hyper-connections (4 residual copies with
  Sinkhorn-mixed `hc_pre` / `hc_post`), hash-routed first layers.
- **Kimi K3** — Kimi Delta Attention on 3 of every 4 layers, output-gated MLA on the rest, attention residuals over
  earlier layers, routed experts at a 3584-d latent width.
- **Qwen3.8 / Qwen3-Next** — gated DeltaNet on 3 of every 4 layers, output-gated GQA on the rest, σ-gated shared expert.

**Not modeled yet**: Nemotron-3.5 (Mamba hybrid), Gemma 4.

## Parallelism conventions

- **Mesh**: `world = dp × cp × tp`, ordered outer → inner (TP innermost / intra-node).
- **TP** (Megatron): `wq/wk/wv`, `q_b_proj`, `kv_b_proj`, `w_gate/w_up`, sinks → column-parallel `Shard(0)`;
  `wo`, `w_down` → row-parallel `Shard(1)`; embeddings / `lm_head` → vocab-parallel `Shard(0)` (toggle);
  norms, router, MLA low-rank `*_a_proj` → replicated. When `n_kv_heads < tp`, KV heads are replicated.
  Grouped experts shard `dim 1` (`w_gate/w_up`) / `dim 2` (`w_down`) when "TP inside experts" is on.
- **SP**: with TP, norm / residual regions shard the sequence over TP (all-gather / reduce-scatter instead of all-reduce).
- **EP**: carved out of the DP axis (`dp % ep == 0`); EP groups are contiguous DP slices. Grouped expert tensors
  `Shard(0)` over EP; ModuleList experts are placed whole on their owning EP rank.
- **FSDP** (FSDP2 semantics): each param's TP/EP-local tensor is `Shard(0)` over the DP group — experts over the
  expert-DP group (`dp / ep`). Uneven sizes use `torch.chunk` rules. Buffers are not sharded.
- **HSDP** (hybrid shard): FSDP within blocks of `hsdpShard` DP ranks, replicated across blocks
  (experts shard over `hsdpShard / ep`).
- **DDP**: replicated; grads all-reduced in 25 MiB buckets.
- **CP**: parameters untouched; the sequence axis is split `seq / cp`, with ring (send/recv KV) or all-gather-KV attention.

## Develop

```bash
npm install
npm run dev        # http://localhost:5173/shardlock/
npm run validate   # sharding invariants: shards tile every tensor across ~7.7k configs, reference param counts
npm run lint
npm run build      # → docs/
```

## Roadmap

- **Pipeline parallelism** — the mesh dims are ordered so a `pp` axis can be prepended; with 2 layers, `PP=2`
  puts one `TransformerBlock` per stage (embeddings on stage 0, `final_norm` / `lm_head` on the last).
