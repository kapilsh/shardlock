import { useStore } from '../store/store.js'
import { useMemo } from 'react'
import { PRESET_GROUPS, PRESETS, fullModelStats } from '../lib/presets.js'
import { formatCount } from '../lib/format.js'
import { NumberField, Section, Segmented, SelectField, Toggle } from './Fields.jsx'

function PresetSelect({ value, onChange }) {
  return (
    <label className="field">
      <span className="field-label">Preset</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {value === 'custom' && <option value="custom">Custom</option>}
        {PRESET_GROUPS.map((g) => (
          <optgroup key={g} label={g}>
            {Object.entries(PRESETS)
              .filter(([, p]) => p.group === g)
              .map(([k, p]) => (
                <option key={k} value={k}>
                  {p.label}
                </option>
              ))}
          </optgroup>
        ))}
      </select>
    </label>
  )
}

function PresetInfo({ presetKey }) {
  const preset = PRESETS[presetKey]
  const stats = useMemo(() => (preset ? fullModelStats(preset) : null), [preset])
  if (!preset?.source || !stats) return null
  return (
    <div className="preset-info">
      <a className="mono" href={`https://huggingface.co/${preset.source}`} target="_blank" rel="noreferrer">
        {preset.source}
      </a>
      <div className="preset-stats mono">
        <span>
          {stats.layers} layers{stats.moeLayers && stats.moeLayers !== stats.layers ? ` (${stats.moeLayers} MoE)` : ''}
          {stats.mtp ? ` + ${stats.mtp} MTP` : ''}
        </span>
        <span>
          {formatCount(stats.total)} total{stats.active !== stats.total ? ` · ${formatCount(stats.active)} active` : ''}
        </span>
      </div>
      <p className="side-note">
        Shown truncated to {preset.config.n_layers} layers{preset.full.n_dense_layers > preset.config.n_dense_layers ? ` (1 of ${preset.full.n_dense_layers} leading dense layers kept)` : ''}.
        {preset.note ? ` Note: ${preset.note}.` : ''}
      </p>
    </div>
  )
}

export default function ModelPanel() {
  const presetKey = useStore((s) => s.presetKey)
  const m = useStore((s) => s.model)
  const applyPreset = useStore((s) => s.applyPreset)
  const set = useStore((s) => s.setModel)
  const moe = m.arch === 'moe'
  const mla = m.attn_type === 'mla'
  const v4 = m.attn_type === 'dsv4'
  const hybrid = !v4 && m.linear_attn !== 'none'

  return (
    <>
      <Section title="Model">
        <PresetSelect value={presetKey} onChange={applyPreset} />
        <PresetInfo presetKey={presetKey} />
        <Segmented value={m.arch} onChange={(arch) => set({ arch })} options={{ dense: 'Dense', moe: 'Sparse (MoE)' }} />
        <div className="grid2">
          <NumberField label="n_layers" value={m.n_layers} min={1} onChange={(v) => set({ n_layers: Math.min(v, 8) })} hint="Kept small on purpose; PP will split these later" />
          <NumberField label="dim" value={m.dim} min={1} onChange={(dim) => set({ dim })} />
          <NumberField label="vocab_size" value={m.vocab_size} min={1} onChange={(vocab_size) => set({ vocab_size })} />
          <Toggle label="tie embeddings" checked={m.tie_embeddings} onChange={(tie_embeddings) => set({ tie_embeddings })} />
        </div>
      </Section>

      <Section title="Attention">
        <Segmented value={m.attn_type} onChange={(attn_type) => set({ attn_type })} options={{ mha: 'MHA', gqa: 'GQA', mla: 'MLA', dsv4: 'DSv4' }} />
        <div className="grid2">
          <NumberField label="n_heads" value={m.n_heads} min={1} onChange={(n_heads) => set({ n_heads })} />
          {v4 && (
            <>
              <NumberField label="head_dim" value={m.head_dim} min={1} onChange={(head_dim) => set({ head_dim })} />
              <NumberField label="rope_head_dim" value={m.qk_rope_head_dim} min={0} onChange={(qk_rope_head_dim) => set({ qk_rope_head_dim })} />
              <NumberField label="q_lora_rank" value={m.q_lora_rank} min={1} onChange={(q_lora_rank) => set({ q_lora_rank })} />
              <NumberField label="o_lora_rank" value={m.o_lora_rank} min={1} onChange={(o_lora_rank) => set({ o_lora_rank })} hint="Grouped low-rank output projection rank" />
              <NumberField label="o_groups" value={m.o_groups} min={1} onChange={(o_groups) => set({ o_groups })} />
              <NumberField label="window" value={m.window_size} min={1} onChange={(window_size) => set({ window_size })} hint="Sliding-window KV length" />
              <NumberField label="indexer heads" value={m.index_n_heads} min={0} onChange={(index_n_heads) => set({ index_n_heads })} hint="Lightning indexer on ×4 compressed layers; 0 = off" />
              <NumberField label="indexer head_dim" value={m.index_head_dim} min={1} onChange={(index_head_dim) => set({ index_head_dim })} />
              <NumberField label="indexer top-k" value={m.index_topk} min={1} onChange={(index_topk) => set({ index_topk })} />
              <NumberField label="hc copies" value={m.hc_mult} min={1} onChange={(hc_mult) => set({ hc_mult })} hint="Hyper-connection residual copies" />
              <NumberField label="hash-routed layers" value={m.n_hash_layers} min={0} onChange={(n_hash_layers) => set({ n_hash_layers })} hint="Leading layers routed by token-id lookup" />
            </>
          )}
          {!mla && !v4 && (
            <>
              <NumberField label="n_kv_heads" value={m.attn_type === 'mha' ? m.n_heads : m.n_kv_heads} min={1} disabled={m.attn_type === 'mha'} onChange={(n_kv_heads) => set({ n_kv_heads })} />
              <NumberField label="head_dim" value={m.head_dim} min={1} onChange={(head_dim) => set({ head_dim })} />
            </>
          )}
          {mla && (
            <>
              <NumberField label="q_lora_rank" value={m.q_lora_rank} min={0} onChange={(q_lora_rank) => set({ q_lora_rank })} hint="0 = direct query projection" />
              <NumberField label="kv_lora_rank" value={m.kv_lora_rank} min={1} onChange={(kv_lora_rank) => set({ kv_lora_rank })} />
              <NumberField label="qk_nope_head_dim" value={m.qk_nope_head_dim} min={0} onChange={(qk_nope_head_dim) => set({ qk_nope_head_dim })} />
              <NumberField label="qk_rope_head_dim" value={m.qk_rope_head_dim} min={0} onChange={(qk_rope_head_dim) => set({ qk_rope_head_dim })} />
              <NumberField label="v_head_dim" value={m.v_head_dim} min={1} onChange={(v_head_dim) => set({ v_head_dim })} />
              <NumberField label="indexer heads" value={m.index_n_heads} min={0} onChange={(index_n_heads) => set({ index_n_heads })} hint="DSA lightning indexer (DeepSeek-V3.2, GLM-5.x); 0 = off" />
              {m.index_n_heads > 0 && (
                <>
                  <NumberField label="indexer head_dim" value={m.index_head_dim} min={1} onChange={(index_head_dim) => set({ index_head_dim })} />
                  <NumberField label="indexer top-k" value={m.index_topk} min={1} onChange={(index_topk) => set({ index_topk })} />
                  <NumberField label="indexer every n" value={m.index_freq} min={1} onChange={(index_freq) => set({ index_freq: Math.max(1, index_freq) })} hint="After the first-k full indexers, one full indexer every n layers (GLM-5.3: 4)" />
                </>
              )}
            </>
          )}
        </div>
        {v4 && (
          <label className="field" title="Per layer: 0 = sliding window only, 4 = compressed ×4 + indexer, 128 = compressed ×128">
            <span className="field-label">KV compress ratio per layer</span>
            <input
              className="mono"
              defaultValue={(m.compress_ratios ?? []).join(', ')}
              key={(m.compress_ratios ?? []).join(',')}
              onBlur={(e) => {
                const ratios = e.target.value.split(/[\s,]+/).filter(Boolean).map(Number).filter((x) => Number.isInteger(x) && x >= 0)
                set({ compress_ratios: ratios })
              }}
            />
          </label>
        )}
        {!mla && !v4 && (
          <div className="toggles">
            <Toggle label="qkv bias" checked={m.qkv_bias} onChange={(qkv_bias) => set({ qkv_bias })} />
            <Toggle label="out bias" checked={m.o_bias} onChange={(o_bias) => set({ o_bias })} />
            <Toggle label="qk norm" checked={m.qk_norm} onChange={(qk_norm) => set({ qk_norm })} />
            {m.qk_norm && (
              <Toggle label="qk norm over all heads" checked={m.qk_norm_type === 'full'} onChange={(full) => set({ qk_norm_type: full ? 'full' : 'head' })} hint="MiniMax-M2 style: one RMSNorm over heads × head_dim" />
            )}
            <Toggle label="attn sinks" checked={m.attn_sinks} onChange={(attn_sinks) => set({ attn_sinks })} />
          </div>
        )}
        <div className="toggles">
          {!v4 && <Toggle label="output gate" checked={m.attn_output_gate} onChange={(attn_output_gate) => set({ attn_output_gate })} hint="Sigmoid gate on the attention output (Qwen3-Next: doubled q proj; MLA: g_proj)" />}
          <Toggle label="attention residuals" checked={m.attn_res} onChange={(attn_res) => set({ attn_res })} hint="Kimi K3: each sublayer input mixes earlier layer outputs" />
        </div>
      </Section>

      {!v4 && (
        <Section title="Linear attention (hybrid)">
          <SelectField
            label="Linear layers"
            value={m.linear_attn}
            onChange={(linear_attn) => set({ linear_attn })}
            options={{ none: 'None (all full attention)', kda: 'Kimi Delta Attention', gdn: 'Gated DeltaNet' }}
          />
          {hybrid && (
            <>
              <div className="grid2">
                <NumberField label="full attn every n" value={m.full_attn_period} min={1} onChange={(full_attn_period) => set({ full_attn_period: Math.max(1, full_attn_period) })} hint="Layers n, 2n, … (1-based) keep the attention type above" />
                <NumberField label={m.linear_attn === 'kda' ? 'heads' : 'value heads'} value={m.la_num_heads} min={1} onChange={(la_num_heads) => set({ la_num_heads })} />
                {m.linear_attn === 'gdn' && <NumberField label="key heads" value={m.la_num_k_heads} min={1} onChange={(la_num_k_heads) => set({ la_num_k_heads })} />}
                <NumberField label={m.linear_attn === 'kda' ? 'head_dim' : 'key head_dim'} value={m.la_head_dim} min={1} onChange={(la_head_dim) => set({ la_head_dim })} />
                {m.linear_attn === 'gdn' && <NumberField label="value head_dim" value={m.la_v_head_dim} min={1} onChange={(la_v_head_dim) => set({ la_v_head_dim })} />}
                <NumberField label="conv kernel" value={m.la_conv} min={1} onChange={(la_conv) => set({ la_conv })} />
              </div>
              <Toggle label="last layer full attention" checked={m.full_attn_last} onChange={(full_attn_last) => set({ full_attn_last })} />
            </>
          )}
        </Section>
      )}

      <Section title={moe ? 'FFN / MoE' : 'FFN'}>
        <div className="grid2">
          {(!moe || m.n_dense_layers > 0) && (
            <NumberField label="ffn_dim (dense)" value={m.ffn_dim} min={1} onChange={(ffn_dim) => set({ ffn_dim })} />
          )}
          {moe && (
            <>
              <NumberField label="moe_inter_dim" value={m.moe_inter_dim} min={1} onChange={(moe_inter_dim) => set({ moe_inter_dim })} />
              <NumberField label="routed experts" value={m.n_routed_experts} min={1} onChange={(n_routed_experts) => set({ n_routed_experts: Math.min(n_routed_experts, 1024) })} />
              <NumberField label="top-k" value={m.n_activated_experts} min={1} onChange={(n_activated_experts) => set({ n_activated_experts })} />
              <NumberField label="shared experts" value={m.n_shared_experts} min={0} onChange={(n_shared_experts) => set({ n_shared_experts })} />
              <NumberField label="first-k dense" value={m.n_dense_layers} min={0} onChange={(n_dense_layers) => set({ n_dense_layers })} hint="Leading layers that use a dense FFN (DeepSeek V3: 3)" />
              <NumberField label="MoE every n layers" value={m.moe_layer_step} min={1} onChange={(moe_layer_step) => set({ moe_layer_step: Math.max(1, moe_layer_step) })} hint="Interleave dense and MoE layers (Llama 4 Maverick: 2)" />
              <NumberField label="latent expert width" value={m.moe_latent_dim} min={0} onChange={(moe_latent_dim) => set({ moe_latent_dim })} hint="Kimi K3: routed experts run at this width behind shared down/up projections; 0 = off" />
            </>
          )}
        </div>
        {moe && (
          <>
            <SelectField
              label="Expert storage"
              value={m.expert_layout}
              onChange={(expert_layout) => set({ expert_layout })}
              options={{ grouped: 'Grouped tensors [E, …]', module_list: 'ModuleList experts.{e}.*' }}
            />
            <div className="toggles">
              <Toggle label="router bias" checked={m.router_bias} onChange={(router_bias) => set({ router_bias })} />
              <Toggle label="expert bias" checked={m.expert_bias} onChange={(expert_bias) => set({ expert_bias })} />
              <Toggle label="balance buffer" checked={m.balance_bias} onChange={(balance_bias) => set({ balance_bias })} />
              {m.n_shared_experts > 0 && <Toggle label="shared expert gate" checked={m.shared_expert_gate} onChange={(shared_expert_gate) => set({ shared_expert_gate })} />}
            </div>
          </>
        )}
        {!moe && <Toggle label="mlp bias" checked={m.mlp_bias} onChange={(mlp_bias) => set({ mlp_bias })} />}
      </Section>
    </>
  )
}
