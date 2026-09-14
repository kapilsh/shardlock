// Formatting helpers shared by every view.

export const DTYPE_BYTES = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
}

export const DTYPES = Object.keys(DTYPE_BYTES)

export const prod = (shape) => shape.reduce((a, b) => a * b, 1)

export function formatShape(shape) {
  if (!shape) return '—'
  return `[${shape.map((d) => d.toLocaleString('en-US')).join(', ')}]`
}

export function formatCount(n) {
  if (n == null) return '—'
  const abs = Math.abs(n)
  if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (abs >= 1e4) return `${(n / 1e3).toFixed(1)}K`
  return n.toLocaleString('en-US')
}

export function formatBytes(b) {
  if (b == null) return '—'
  const abs = Math.abs(b)
  if (abs >= 2 ** 40) return `${(b / 2 ** 40).toFixed(2)} TiB`
  if (abs >= 2 ** 30) return `${(b / 2 ** 30).toFixed(2)} GiB`
  if (abs >= 2 ** 20) return `${(b / 2 ** 20).toFixed(2)} MiB`
  if (abs >= 2 ** 10) return `${(b / 2 ** 10).toFixed(1)} KiB`
  return `${b} B`
}
