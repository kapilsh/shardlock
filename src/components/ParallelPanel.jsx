import { useStore } from '../store/store.js'
import { DP_STRATEGIES, worldSize } from '../lib/mesh.js'
import { DTYPES } from '../lib/format.js'
import { OPTIMIZERS } from '../lib/memory.js'
import { DegreeField, NumberField, Section, Segmented, SelectField, Toggle } from './Fields.jsx'

const dtypeOptions = Object.fromEntries(DTYPES.map((d) => [d, d]))

export default function ParallelPanel() {
  const p = useStore((s) => s.parallel)
  const model = useStore((s) => s.model)
  const prec = useStore((s) => s.precision)
  const train = useStore((s) => s.train)
  const set = useStore((s) => s.setParallel)
  const setPrec = useStore((s) => s.setPrecision)
  const setTrain = useStore((s) => s.setTrain)
  const moe = model.arch === 'moe'

  return (
    <>
      <Section title="Parallelism" aside={<span className="mono world">world = {worldSize(p).toLocaleString('en-US')}</span>}>
        <Segmented value={p.dpStrategy} onChange={(dpStrategy) => set({ dpStrategy })} options={DP_STRATEGIES} />
        <div className="degrees">
          <DegreeField label="DP" color="var(--ax-dp)" value={p.dp} onChange={(dp) => set({ dp })} hint="Data-parallel degree (DDP replicas / FSDP shards)" />
          <DegreeField label="TP" color="var(--ax-tp)" value={p.tp} onChange={(tp) => set({ tp })} hint="Tensor parallel (innermost mesh axis)" />
          <DegreeField label="CP" color="var(--ax-cp)" value={p.cp} onChange={(cp) => set({ cp })} hint="Context parallel: shards the sequence" />
          <DegreeField label="EP" color="var(--ax-ep)" value={p.ep} onChange={(ep) => set({ ep })} hint="Expert parallel, carved out of the DP axis" />
          {p.dpStrategy === 'hsdp' && (
            <DegreeField label="HSDP shard" color="var(--ax-hsdp)" value={p.hsdpShard} onChange={(hsdpShard) => set({ hsdpShard })} hint="FSDP shard group size in DP ranks (typically one node); replicated across groups" />
          )}
        </div>
        <div className="toggles">
          <Toggle label="sequence parallel" checked={p.sp} disabled={p.tp <= 1} onChange={(sp) => set({ sp })} hint="Megatron SP: shard norms/residual activations over TP" />
          <Toggle label="vocab-parallel emb/head" checked={p.vocabParallel} disabled={p.tp <= 1} onChange={(vocabParallel) => set({ vocabParallel })} />
          {moe && <Toggle label="TP inside experts" checked={p.tpExperts} disabled={p.tp <= 1} onChange={(tpExperts) => set({ tpExperts })} />}
          {p.dpStrategy !== 'ddp' && (
            <Toggle label="reshard after forward" checked={p.reshardAfterForward} onChange={(reshardAfterForward) => set({ reshardAfterForward })} />
          )}
        </div>
        {p.cp > 1 && (
          <SelectField label="CP attention" value={p.cpStyle} onChange={(cpStyle) => set({ cpStyle })} options={{ ring: 'Ring attention (send/recv KV)', allgather: 'All-gather KV (Llama 3 style)' }} />
        )}
        <p className="side-note">
          Mesh order <code>[dp, cp, tp]</code> outer → inner. EP groups are contiguous slices of each DP axis. PP is not modeled yet.
        </p>
      </Section>

      <Section title="Precision & optimizer">
        <div className="grid2">
          <SelectField label="weight storage" value={prec.weight} onChange={(weight) => setPrec({ weight })} options={dtypeOptions} />
          <SelectField label="compute / gather" value={prec.compute} onChange={(compute) => setPrec({ compute })} options={dtypeOptions} />
          <SelectField label="grads" value={prec.grad} onChange={(grad) => setPrec({ grad })} options={dtypeOptions} />
          <SelectField label="optim states" value={prec.optimDtype} onChange={(optimDtype) => setPrec({ optimDtype })} options={dtypeOptions} />
        </div>
        <SelectField label="optimizer" value={prec.optimizer} onChange={(optimizer) => setPrec({ optimizer })} options={Object.fromEntries(Object.entries(OPTIMIZERS).map(([k, v]) => [k, v.label]))} />
        <Toggle label="fp32 master weights" checked={prec.master} disabled={prec.weight === 'fp32'} onChange={(master) => setPrec({ master })} />
      </Section>

      <Section title="Micro-batch">
        <div className="grid2">
          <NumberField label="batch" value={train.micro_batch} min={1} onChange={(micro_batch) => setTrain({ micro_batch })} />
          <NumberField label="seq_len" value={train.seq_len} min={1} onChange={(seq_len) => setTrain({ seq_len })} />
        </div>
      </Section>
    </>
  )
}
