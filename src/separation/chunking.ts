/**
 * Splitting a long signal into overlapping chunks and joining the per-chunk results with a cross-fade.
 *
 * Neighbouring chunks overlap by `overlap` samples. Each chunk result is weighted by a raised-cosine ramp
 * over its overlap with the previous and the next chunk, then overlap-added and divided by the sum of the
 * weights, so the join is exact whatever the chunk boundaries are.
 *   equal-gain : weights sin²/cos² (sum to 1; right for two estimates of the same signal)
 *   equal-power: weights sin/cos (power sums to 1; right for uncorrelated signals)
 */
export type Crossfade = 'equal-gain' | 'equal-power';

export interface ChunkRange {
  start: number;
  end: number;
}

/**
 * Splits `length` samples into the fewest chunks of at most `chunkLength` samples that overlap by at least
 * `overlap`; the chunks are of (nearly) equal length so no work is wasted on a short or heavily overlapping
 * last chunk.
 */
export function planChunks(length: number, chunkLength: number, overlap: number): ChunkRange[] {
  if (chunkLength <= 0) throw new Error('planChunks: chunkLength must be positive');
  if (overlap < 0 || overlap >= chunkLength) throw new Error('planChunks: overlap must be in [0, chunkLength)');
  if (length <= chunkLength) return [{ start: 0, end: length }];
  const n = Math.ceil((length - overlap) / (chunkLength - overlap));
  const len = Math.min(chunkLength, Math.ceil((length + (n - 1) * overlap) / n));
  const stride = len - overlap;
  const ranges: ChunkRange[] = [];
  for (let i = 0; i < n; i++) {
    const start = i === n - 1 ? length - len : Math.min(i * stride, length - len);
    ranges.push({ start, end: start + len });
  }
  return ranges;
}

function ramp(i: number, n: number, mode: Crossfade): number {
  // i in [0, n): 0 -> 1
  const s = Math.sin((Math.PI / 2) * ((i + 0.5) / n));
  return mode === 'equal-gain' ? s * s : s;
}

/** Accumulates chunk results into one signal; call `add` for every chunk in `ranges`, then `finish`. */
export class ChunkAssembler {
  private readonly acc: Float64Array;
  private readonly weight: Float64Array;
  private readonly ranges: ChunkRange[];
  private readonly mode: Crossfade;

  constructor(length: number, ranges: ChunkRange[], mode: Crossfade) {
    this.acc = new Float64Array(length);
    this.weight = new Float64Array(length);
    this.ranges = ranges;
    this.mode = mode;
  }

  add(index: number, data: Float32Array): void {
    const { start, end } = this.ranges[index];
    const n = end - start;
    if (data.length !== n) throw new Error(`ChunkAssembler: chunk ${index} has ${data.length} samples, expected ${n}`);
    const prev = index > 0 ? this.ranges[index - 1] : null;
    const next = index < this.ranges.length - 1 ? this.ranges[index + 1] : null;
    const rampIn = prev ? Math.max(0, Math.min(n, prev.end - start)) : 0;
    const rampOut = next ? Math.max(0, Math.min(n, end - next.start)) : 0;
    for (let i = 0; i < n; i++) {
      let w = 1;
      if (i < rampIn) w *= ramp(i, rampIn, this.mode);
      if (i >= n - rampOut) w *= ramp(n - 1 - i, rampOut, this.mode);
      this.acc[start + i] += w * data[i];
      this.weight[start + i] += w;
    }
  }

  finish(out?: Float32Array): Float32Array {
    const y = out ?? new Float32Array(this.acc.length);
    for (let i = 0; i < y.length; i++) {
      const w = this.weight[i];
      y[i] = w > 0 ? this.acc[i] / w : 0;
    }
    return y;
  }
}
