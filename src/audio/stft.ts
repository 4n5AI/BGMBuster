/**
 * STFT / iSTFT with the exact conventions of MERL's MRX (`torch.stft` / `torch.istft`):
 *   - periodic Hann window of length n_fft, hop 256
 *   - `normalized=True`  : spectrum scaled by 1 / sqrt(n_fft)
 *   - `center=True`      : the signal is padded by n_fft / 2 on both sides with reflection
 *   - frames = 1 + floor(length / hop)
 *   - iSTFT: overlap-add of window * frame, divided by the window-square envelope (NOLA), cut to `length`
 *
 * Spectrograms are stored frame-major: index = t * bins + f (so one frame is contiguous).
 */
import { RealFFT } from './fft.ts';

export interface Spectrogram {
  readonly nFft: number;
  readonly hop: number;
  readonly bins: number;
  readonly frames: number;
  readonly length: number; // original signal length
  readonly re: Float32Array; // [frames * bins]
  readonly im: Float32Array; // [frames * bins]
}

const windowCache = new Map<number, Float32Array>();

/** Periodic Hann window (torch.hann_window default). */
export function hannPeriodic(n: number): Float32Array {
  let w = windowCache.get(n);
  if (!w) {
    w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    windowCache.set(n, w);
  }
  return w;
}

export function frameCount(length: number, hop: number): number {
  return 1 + Math.floor(length / hop);
}

/** Reflect-padded sample access (torch "reflect": x[-1] = x[1], x[L] = x[L-2]). */
function reflectIndex(i: number, length: number): number {
  if (i < 0) i = -i;
  if (i >= length) i = 2 * (length - 1) - i;
  return i;
}

export function stft(signal: Float32Array, nFft: number, hop: number): Spectrogram {
  const length = signal.length;
  if (length < 2) throw new Error('stft: signal too short');
  const bins = nFft / 2 + 1;
  const frames = frameCount(length, hop);
  const pad = nFft >> 1;
  const win = hannPeriodic(nFft);
  const fft = RealFFT.get(nFft);
  const scale = 1 / Math.sqrt(nFft);
  const frame = new Float64Array(nFft);
  const re = new Float32Array(frames * bins);
  const im = new Float32Array(frames * bins);
  const fr = new Float64Array(bins);
  const fi = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const start = t * hop - pad;
    for (let i = 0; i < nFft; i++) {
      let idx = start + i;
      if (idx < 0 || idx >= length) idx = reflectIndex(idx, length);
      frame[i] = signal[idx] * win[i];
    }
    fft.forward(frame, fr, fi);
    const base = t * bins;
    for (let f = 0; f < bins; f++) {
      re[base + f] = fr[f] * scale;
      im[base + f] = fi[f] * scale;
    }
  }
  return { nFft, hop, bins, frames, length, re, im };
}

/**
 * Magnitudes of a spectrogram laid out bin-major ([bins, frames], index = f * frames + t), which is the
 * [F, T] layout the ONNX graph takes for one batch item.
 */
export function magnitudeFT(spec: Spectrogram, out?: Float32Array): Float32Array {
  const { bins, frames, re, im } = spec;
  const mag = out ?? new Float32Array(bins * frames);
  for (let t = 0; t < frames; t++) {
    const base = t * bins;
    for (let f = 0; f < bins; f++) {
      const r = re[base + f];
      const i = im[base + f];
      mag[f * frames + t] = Math.sqrt(r * r + i * i);
    }
  }
  return mag;
}

/**
 * iSTFT. `maskFT` (optional, layout [bins, frames]) multiplies the spectrogram before inversion.
 * Adds the result into `out` (so the resolutions can be summed in place) and returns it.
 */
export function istft(spec: Spectrogram, maskFT?: Float32Array, out?: Float32Array): Float32Array {
  const { nFft, hop, bins, frames, length, re, im } = spec;
  const pad = nFft >> 1;
  const win = hannPeriodic(nFft);
  const fft = RealFFT.get(nFft);
  const scale = Math.sqrt(nFft); // undo normalized=True
  const result = out ?? new Float32Array(length);
  if (result.length !== length) throw new Error('istft: output length mismatch');
  const total = nFft + hop * (frames - 1);
  const acc = new Float64Array(total);
  const env = windowSquareEnvelope(nFft, hop, frames);
  const fr = new Float64Array(bins);
  const fi = new Float64Array(bins);
  const frame = new Float64Array(nFft);
  for (let t = 0; t < frames; t++) {
    const base = t * bins;
    if (maskFT) {
      for (let f = 0; f < bins; f++) {
        const m = maskFT[f * frames + t] * scale;
        fr[f] = re[base + f] * m;
        fi[f] = im[base + f] * m;
      }
    } else {
      for (let f = 0; f < bins; f++) {
        fr[f] = re[base + f] * scale;
        fi[f] = im[base + f] * scale;
      }
    }
    fft.inverse(fr, fi, frame);
    const off = t * hop;
    for (let i = 0; i < nFft; i++) acc[off + i] += frame[i] * win[i];
  }
  for (let n = 0; n < length; n++) {
    const e = env[pad + n];
    result[n] += e > 1e-11 ? acc[pad + n] / e : 0;
  }
  return result;
}

const envCache = new Map<string, Float64Array>();

function windowSquareEnvelope(nFft: number, hop: number, frames: number): Float64Array {
  const key = `${nFft}/${hop}/${frames}`;
  let env = envCache.get(key);
  if (env) return env;
  const win = hannPeriodic(nFft);
  env = new Float64Array(nFft + hop * (frames - 1));
  for (let t = 0; t < frames; t++) {
    const off = t * hop;
    for (let i = 0; i < nFft; i++) env[off + i] += win[i] * win[i];
  }
  if (envCache.size > 8) envCache.clear();
  envCache.set(key, env);
  return env;
}
