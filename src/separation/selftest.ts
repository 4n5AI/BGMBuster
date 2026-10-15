/**
 * Backend self-test: a fixed one-second probe is run through the model and a few mask statistics are compared
 * with values computed by the WebAssembly backend (which matches the upstream PyTorch implementation). A GPU
 * backend whose kernels misbehave (silently wrong LSTM output, all-zero masks) fails this and the app falls back
 * to WebAssembly. Reference values live in config.ts (SELFTEST).
 */
import type { Spectrogram } from '../audio/stft.ts';

export const SELFTEST_SECONDS = 1;
export const SELFTEST_SAMPLE_RATE = 44100;

/** Deterministic probe: a chord, an amplitude-modulated harmonic "voice" and noise bursts, at about -27 LUFS. */
export function probeSignal(): Float32Array {
  const sr = SELFTEST_SAMPLE_RATE;
  const n = sr * SELFTEST_SECONDS;
  const x = new Float32Array(n);
  let state = 20240607;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    state = (state * 1664525 + 1013904223) >>> 0;
    const noise = (state / 4294967296 - 0.5) * (Math.sin(2 * Math.PI * 3 * t) > 0.6 ? 0.08 : 0);
    const chord = 0.05 * (Math.sin(2 * Math.PI * 220 * t) + 0.6 * Math.sin(2 * Math.PI * 277.2 * t) + 0.5 * Math.sin(2 * Math.PI * 329.6 * t));
    const voice = 0.06 * Math.sin(2 * Math.PI * 140 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 5 * t)) * (1 + 0.6 * Math.sin(2 * Math.PI * 900 * t));
    x[i] = chord + voice + noise;
  }
  return x;
}

export interface MaskStats {
  /** Mean mask value per source for mask_1024 and mask_8192: [music, speech, sfx]. */
  m1024: [number, number, number];
  m8192: [number, number, number];
}

interface TensorLike {
  /** onnxruntime's Tensor.data (a typed array for float outputs). */
  data: unknown;
  dims: readonly number[];
}

export function maskStats(outputs: Record<string, TensorLike>, specs: readonly Spectrogram[]): MaskStats {
  const mean = (name: string, spec: Spectrogram): [number, number, number] => {
    const data = outputs[name].data as ArrayLike<number>;
    const plane = spec.bins * spec.frames;
    const out: [number, number, number] = [0, 0, 0];
    for (let s = 0; s < 3; s++) {
      let acc = 0;
      for (let i = s * plane; i < (s + 1) * plane; i++) acc += data[i];
      out[s] = acc / plane;
    }
    return out;
  };
  return { m1024: mean('mask_1024', specs[0]), m8192: mean('mask_8192', specs[2]) };
}

/** True when `got` is within `tolerance` (relative, with a small absolute floor) of `expected` for every value. */
export function statsMatch(got: MaskStats, expected: MaskStats, tolerance = 0.05): boolean {
  const pairs = [...got.m1024.map((v, i) => [v, expected.m1024[i]]), ...got.m8192.map((v, i) => [v, expected.m8192[i]])];
  return pairs.every(([a, b]) => Number.isFinite(a) && Math.abs(a - b) <= tolerance * Math.max(Math.abs(b), 0.01));
}
