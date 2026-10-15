/**
 * ITU-R BS.1770-4 integrated loudness, implemented to match pyloudnorm's default "K-weighting" meter
 * (RBJ biquads: high shelf +4 dB at 1500 Hz, Q 1/sqrt 2; high-pass at 38 Hz, Q 0.5; 400 ms blocks with
 * 75 % overlap; absolute gate -70 LUFS, relative gate -10 LU). Upstream MRX (separate.py) uses that meter to
 * bring the input to -27 LUFS, so the gain computed here reproduces upstream's.
 */

interface Biquad {
  b0: number; b1: number; b2: number; a1: number; a2: number;
}

/** RBJ audio EQ cookbook designs, normalised by a0 (what scipy.signal.lfilter does with pyloudnorm's a[0]). */
function rbjHighShelf(sampleRate: number, gainDb: number, q: number, fc: number): Biquad {
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * fc) / sampleRate;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const sqA = Math.sqrt(A);
  const a0 = A + 1 - (A - 1) * cos + 2 * sqA * alpha;
  return {
    b0: (A * (A + 1 + (A - 1) * cos + 2 * sqA * alpha)) / a0,
    b1: (-2 * A * (A - 1 + (A + 1) * cos)) / a0,
    b2: (A * (A + 1 + (A - 1) * cos - 2 * sqA * alpha)) / a0,
    a1: (2 * (A - 1 - (A + 1) * cos)) / a0,
    a2: (A + 1 - (A - 1) * cos - 2 * sqA * alpha) / a0,
  };
}

function rbjHighPass(sampleRate: number, q: number, fc: number): Biquad {
  const w0 = (2 * Math.PI * fc) / sampleRate;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cos) / 2 / a0,
    b1: -(1 + cos) / a0,
    b2: (1 + cos) / 2 / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
  };
}

function filter(x: ArrayLike<number>, f: Biquad): Float64Array {
  const y = new Float64Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let n = 0; n < x.length; n++) {
    const x0 = x[n];
    const y0 = f.b0 * x0 + f.b1 * x1 + f.b2 * x2 - f.a1 * y1 - f.a2 * y2;
    y[n] = y0;
    x2 = x1; x1 = x0; y2 = y1; y1 = y0;
  }
  return y;
}

/** Channel weights G for up to five channels (L, R, C, Ls, Rs); more channels get 1. */
function channelWeight(index: number): number {
  return index === 3 || index === 4 ? 1.41 : 1.0;
}

const BLOCK_SECONDS = 0.4;
const STEP_FRACTION = 0.25; // 1 - 75 % overlap

/**
 * Integrated loudness in LUFS of a multichannel signal. Returns -Infinity for silence
 * (or anything shorter than one 400 ms block).
 */
export function integratedLoudness(channels: readonly Float32Array[], sampleRate: number): number {
  const length = channels[0]?.length ?? 0;
  const seconds = length / sampleRate;
  if (seconds < BLOCK_SECONDS) return -Infinity;
  const nBlocks = Math.round((seconds - BLOCK_SECONDS) / (BLOCK_SECONDS * STEP_FRACTION)) + 1;
  const shelf = rbjHighShelf(sampleRate, 4.0, Math.SQRT1_2, 1500.0);
  const hp = rbjHighPass(sampleRate, 0.5, 38.0);
  const blockSamples = BLOCK_SECONDS * sampleRate;
  // z[c][j]: mean square of block j in channel c (K-weighted), block bounds as pyloudnorm computes them
  const z: Float64Array[] = channels.map((ch) => {
    const y = filter(filter(ch, shelf), hp);
    const out = new Float64Array(nBlocks);
    for (let j = 0; j < nBlocks; j++) {
      const lo = Math.floor(BLOCK_SECONDS * (j * STEP_FRACTION) * sampleRate);
      const hi = Math.min(length, Math.floor(BLOCK_SECONDS * (j * STEP_FRACTION + 1) * sampleRate));
      let acc = 0;
      for (let i = lo; i < hi; i++) acc += y[i] * y[i];
      out[j] = acc / blockSamples;
    }
    return out;
  });
  const blockLoudness = new Float64Array(nBlocks);
  for (let j = 0; j < nBlocks; j++) {
    let sum = 0;
    for (let c = 0; c < z.length; c++) sum += channelWeight(c) * z[c][j];
    blockLoudness[j] = -0.691 + 10 * Math.log10(sum);
  }
  const gatedLoudness = (keep: (l: number) => boolean): number => {
    let count = 0;
    const means = new Float64Array(z.length);
    for (let j = 0; j < nBlocks; j++) {
      if (keep(blockLoudness[j])) {
        count++;
        for (let c = 0; c < z.length; c++) means[c] += z[c][j];
      }
    }
    if (count === 0) return -Infinity;
    let sum = 0;
    for (let c = 0; c < z.length; c++) sum += (channelWeight(c) * means[c]) / count;
    return -0.691 + 10 * Math.log10(sum);
  };
  const absolute = -70;
  const afterAbsolute = gatedLoudness((l) => l >= absolute);
  if (!Number.isFinite(afterAbsolute)) return -Infinity;
  const relative = afterAbsolute - 10;
  const result = gatedLoudness((l) => l > relative && l > absolute);
  return Number.isFinite(result) ? result : -Infinity;
}

/** Linear gain that brings `loudness` to `targetLufs`; 1 for silence. */
export function gainToTarget(loudness: number, targetLufs: number): number {
  if (!Number.isFinite(loudness)) return 1;
  return Math.pow(10, (targetLufs - loudness) / 20);
}
