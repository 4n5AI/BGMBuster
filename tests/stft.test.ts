import { describe, expect, it } from 'vitest';
import { RealFFT } from '../src/audio/fft.ts';
import { frameCount, istft, magnitudeFT, stft } from '../src/audio/stft.ts';

function snrDb(est: ArrayLike<number>, ref: ArrayLike<number>): number {
  let s = 0;
  let e = 0;
  for (let i = 0; i < ref.length; i++) {
    s += ref[i] * ref[i];
    e += (est[i] - ref[i]) ** 2;
  }
  return 10 * Math.log10(s / (e + 1e-30));
}

function testSignal(length: number, seed = 1): Float32Array {
  const x = new Float32Array(length);
  let state = seed;
  for (let i = 0; i < length; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const noise = (state / 4294967296 - 0.5) * 0.1;
    x[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / 44100) + 0.2 * Math.sin((2 * Math.PI * 3001 * i) / 44100 + 0.3) + noise;
  }
  return x;
}

describe('RealFFT', () => {
  it('matches a direct DFT and inverts exactly', () => {
    const n = 64;
    const fft = RealFFT.get(n);
    const x = testSignal(n, 7);
    const re = new Float64Array(n / 2 + 1);
    const im = new Float64Array(n / 2 + 1);
    fft.forward(x, re, im);
    for (let k = 0; k <= n / 2; k++) {
      let dr = 0;
      let di = 0;
      for (let i = 0; i < n; i++) {
        const a = (-2 * Math.PI * k * i) / n;
        dr += x[i] * Math.cos(a);
        di += x[i] * Math.sin(a);
      }
      expect(re[k]).toBeCloseTo(dr, 9);
      expect(im[k]).toBeCloseTo(di, 9);
    }
    const y = new Float64Array(n);
    fft.inverse(re, im, y);
    expect(snrDb(y, x)).toBeGreaterThan(200);
  });
});

describe('stft / istft', () => {
  it('uses torch.stft frame counts', () => {
    expect(frameCount(88200, 256)).toBe(345);
    expect(stft(testSignal(1000), 1024, 256).frames).toBe(frameCount(1000, 256));
  });

  it('round-trips to better than 120 dB for all three resolutions', () => {
    const x = testSignal(44100);
    for (const nFft of [1024, 2048, 8192]) {
      const spec = stft(x, nFft, 256);
      expect(spec.bins).toBe(nFft / 2 + 1);
      const y = istft(spec);
      expect(snrDb(y, x)).toBeGreaterThan(120);
    }
  });

  it('places a pure tone in the expected bin with the normalized scale', () => {
    const nFft = 1024;
    const sr = 44100;
    const bin = 100;
    const freq = (bin * sr) / nFft;
    const x = new Float32Array(sr);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * freq * i) / sr);
    const spec = stft(x, nFft, 256);
    const mag = magnitudeFT(spec);
    const frames = spec.frames;
    const t = Math.floor(frames / 2);
    let best = 0;
    for (let f = 0; f < spec.bins; f++) if (mag[f * frames + t] > mag[best * frames + t]) best = f;
    expect(best).toBe(bin);
    // |X| = (amplitude * sum(window) / 2) / sqrt(n_fft) = (1 * 512 / 2) / 32 = 8
    expect(mag[bin * frames + t]).toBeCloseTo(8, 3);
  });

  it('applies a bin-major mask in istft', () => {
    const x = testSignal(20000);
    const spec = stft(x, 1024, 256);
    const mask = new Float32Array(spec.bins * spec.frames).fill(0.5);
    const y = istft(spec, mask);
    const half = x.map((v) => v * 0.5);
    expect(snrDb(y, half)).toBeGreaterThan(120);
  });
});
