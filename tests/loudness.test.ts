import { describe, expect, it } from 'vitest';
import { gainToTarget, integratedLoudness } from '../src/audio/loudness.ts';

const sr = 44100;

function tone(): Float32Array {
  const x = new Float32Array(2 * sr);
  for (let i = 0; i < x.length; i++) {
    const t = i / sr;
    x[i] = 0.5 * Math.sin(2 * Math.PI * 440 * t) + 0.2 * Math.sin(2 * Math.PI * 3001 * t + 0.3);
  }
  return x;
}

// Reference values computed with pyloudnorm 0.2.0 (the meter upstream MRX uses) on the same signals.
describe('integratedLoudness (BS.1770, pyloudnorm-compatible)', () => {
  it('matches pyloudnorm for a mono two-tone signal', () => {
    expect(integratedLoudness([tone()], sr)).toBeCloseTo(-8.3454, 2);
  });

  it('matches pyloudnorm for stereo with the right channel at half level', () => {
    const l = tone();
    const r = l.map((v) => v * 0.5);
    expect(integratedLoudness([l, r], sr)).toBeCloseTo(-7.3763, 2);
  });

  it('gates out near-silent blocks', () => {
    const x = tone();
    for (let i = sr; i < x.length; i++) x[i] *= 1e-4;
    expect(integratedLoudness([x], sr)).toBeCloseTo(-9.0511, 2);
  });

  it('returns -Infinity for silence and gain 1', () => {
    const l = integratedLoudness([new Float32Array(sr)], sr);
    expect(l).toBe(-Infinity);
    expect(gainToTarget(l, -27)).toBe(1);
  });

  it('computes the gain to -27 LUFS', () => {
    expect(gainToTarget(-7, -27)).toBeCloseTo(0.1, 6);
  });
});
