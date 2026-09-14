import { useStore } from '../store/store.js'
import { PRESETS } from '../lib/presets.js'
import { NumberField, Section, Segmented, SelectField, Toggle } from './Fields.jsx'

const presetOptions = Object.fromEntries(Object.entries(PRESETS).map(([k, v]) => [k, v.label]))

export default function ModelPanel() {
  const presetKey = useStore((s) => s.presetKey)
  const m = useStore((s) => s.model)
  const applyPreset = useStore((s) => s.applyPreset)
  const set = useStore((s) => s.setModel)
  const moe = m.arch === 'moe'
  const mla = m.attn_type === 'mla'

  return (
    <>
      <Section title="Model">
        <SelectField
          label="Preset"
          value={presetKey}
          onChange={applyPreset}
          options={presetKey === 'custom' ? { custom: 'Custom', ...presetOptions } : presetOptions}
        />
        <Segmented value={m.arch} onChange={(arch) => set({ arch })} options={{ dense: 'Dense', moe: 'Sparse (MoE)' }} />
        <div className="grid2">
          <NumberField label="n_layers" value={m.n_layers} min={1} onChange={(v) => set({ n_layers: Math.min(v, 8) })} hint="Kept small on purpose; PP will split these later" />
          <NumberField label="dim" value={m.dim} min={1} onChange={(dim) => set({ dim })} />
          <NumberField label="vocab_size" value={m.vocab_size} min={1} onChange={(vocab_size) => set({ vocab_size })} />
          <Toggle label="tie embeddings" checked={m.tie_embeddings} onChange={(tie_embeddings) => set({ tie_embeddings })} />
        </div>
      </Section>

      <Section title="Attention">
        <Segmented value={m.attn_type} onChange={(attn_type) => set({ attn_type })} options={{ mha: 'MHA', gqa: 'GQA', mla: 'MLA' }} />
        <div className="grid2">
          <NumberField label="n_heads" value={m.n_heads} min={1} onChange={(n_heads) => set({ n_heads })} />
          {!mla && (
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
            </>
          )}
        </div>
        {!mla && (
          <div className="toggles">
            <Toggle label="qkv bias" checked={m.qkv_bias} onChange={(qkv_bias) => set({ qkv_bias })} />
            <Toggle label="out bias" checked={m.o_bias} onChange={(o_bias) => set({ o_bias })} />
            <Toggle label="qk norm" checked={m.qk_norm} onChange={(qk_norm) => set({ qk_norm })} />
            <Toggle label="attn sinks" checked={m.attn_sinks} onChange={(attn_sinks) => set({ attn_sinks })} />
          </div>
        )}
      </Section>

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
            </div>
          </>
        )}
        {!moe && <Toggle label="mlp bias" checked={m.mlp_bias} onChange={(mlp_bias) => set({ mlp_bias })} />}
      </Section>
    </>
  )
}
