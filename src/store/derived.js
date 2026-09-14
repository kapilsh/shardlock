import { useMemo } from 'react'
import { useStore } from './store.js'
import { buildModel, validateModel } from '../lib/model.js'
import { rankCoords, validateParallel, worldSize } from '../lib/mesh.js'

// Everything computed from the raw config, memoized once per change.
export function useDerived() {
  const model = useStore((s) => s.model)
  const parallel = useStore((s) => s.parallel)
  const train = useStore((s) => s.train)
  const rank = useStore((s) => s.rank)

  const params = useMemo(() => buildModel(model), [model])
  const modelCheck = useMemo(() => validateModel(model), [model])
  const parCheck = useMemo(() => validateParallel(parallel, model, train), [parallel, model, train])
  const world = worldSize(parallel)
  const coords = useMemo(() => rankCoords(parallel, Math.min(rank, world - 1)), [parallel, rank, world])

  const errors = [...modelCheck.errors, ...parCheck.errors]
  const warnings = [...modelCheck.warnings, ...parCheck.warnings]
  return { params, world, coords, errors, warnings, valid: errors.length === 0 }
}
