import { describe, expect, it } from 'vitest';
import { ChunkAssembler, planChunks } from '../src/separation/chunking.ts';

describe('planChunks', () => {
  it('covers the whole signal with equal chunks of at most the requested length and at least the overlap', () => {
    for (const [length, chunk, overlap] of [
      [100, 30, 10],
      [40 * 44100, 20 * 44100, 2 * 44100],
      [61 * 44100, 20 * 44100, 2 * 44100],
      [1000, 999, 0],
    ] as const) {
      const ranges = planChunks(length, chunk, overlap);
      expect(ranges[0].start).toBe(0);
      expect(ranges[ranges.length - 1].end).toBe(length);
      for (const r of ranges) expect(r.end - r.start).toBeLessThanOrEqual(chunk);
      for (let i = 1; i < ranges.length; i++) expect(ranges[i - 1].end - ranges[i].start).toBeGreaterThanOrEqual(overlap);
      const lengths = ranges.map((r) => r.end - r.start);
      expect(Math.max(...lengths) - Math.min(...lengths)).toBeLessThanOrEqual(1);
      // fewest chunks: one fewer would not cover the signal
      const n = ranges.length;
      expect((n - 1) * (chunk - overlap) + overlap).toBeLessThan(length + 1);
      expect((n - 2) * (chunk - overlap) + overlap).toBeLessThan(length);
    }
    const r = planChunks(40 * 44100, 20 * 44100, 2 * 44100);
    expect(r).toHaveLength(3);
    expect(r[0].end - r[0].start).toBe(Math.ceil((44 * 44100) / 3));
  });

  it('returns one chunk for short signals', () => {
    expect(planChunks(10, 30, 10)).toEqual([{ start: 0, end: 10 }]);
    expect(planChunks(30, 30, 10)).toEqual([{ start: 0, end: 30 }]);
  });
});

describe('ChunkAssembler', () => {
  for (const mode of ['equal-gain', 'equal-power'] as const) {
    it(`reassembles identical chunk results without seams (${mode})`, () => {
      const length = 1000;
      const signal = new Float32Array(length);
      for (let i = 0; i < length; i++) signal[i] = Math.sin(i * 0.05) + 0.3 * Math.cos(i * 0.31);
      const ranges = planChunks(length, 300, 100);
      const asm = new ChunkAssembler(length, ranges, mode);
      ranges.forEach((r, k) => asm.add(k, signal.slice(r.start, r.end)));
      const y = asm.finish();
      let maxErr = 0;
      for (let i = 0; i < length; i++) maxErr = Math.max(maxErr, Math.abs(y[i] - signal[i]));
      expect(maxErr).toBeLessThan(1e-6);
    });
  }

  it('blends a step between neighbouring chunks smoothly (equal-gain)', () => {
    const length = 300;
    const ranges = planChunks(length, 200, 100); // [0,200) and [100,300)
    const asm = new ChunkAssembler(length, ranges, 'equal-gain');
    asm.add(0, new Float32Array(200).fill(0));
    asm.add(1, new Float32Array(200).fill(1));
    const y = asm.finish();
    expect(y[0]).toBe(0);
    expect(y[99]).toBe(0);
    expect(y[200]).toBe(1);
    expect(y[299]).toBe(1);
    // monotonic ramp across the overlap, close to 0.5 in the middle
    for (let i = 101; i < 200; i++) expect(y[i]).toBeGreaterThanOrEqual(y[i - 1]);
    expect(y[150]).toBeGreaterThan(0.45);
    expect(y[150]).toBeLessThan(0.55);
  });
});
