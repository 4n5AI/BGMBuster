import { describe, expect, it } from 'vitest';
import { encodeWav16, readWavInfo } from '../src/audio/wav.ts';

describe('encodeWav16', () => {
  it('writes a valid 44-byte RIFF header and interleaved 16-bit samples', () => {
    const left = new Float32Array([0, 0.5, -0.5, 1, -1, 2]);
    const right = new Float32Array([0.25, -0.25, 0, 0, 0, 0]);
    const buffer = encodeWav16([left, right], 48000);
    expect(buffer.byteLength).toBe(44 + 6 * 2 * 2);
    const info = readWavInfo(buffer);
    expect(info).toEqual({ channels: 2, sampleRate: 48000, bitsPerSample: 16, frames: 6 });
    const view = new DataView(buffer);
    expect(view.getUint32(4, true)).toBe(buffer.byteLength - 8);
    expect(view.getUint32(28, true)).toBe(48000 * 4); // byte rate
    expect(view.getUint16(32, true)).toBe(4); // block align
    const pcm = new Int16Array(buffer, 44);
    expect(Array.from(pcm.slice(0, 4))).toEqual([0, 8192, Math.round(0.5 * 32767), Math.round(-0.25 * 32768)]);
    expect(pcm[6]).toBe(32767); // +1.0 clamps to max
    expect(pcm[8]).toBe(-32768); // -1.0
    expect(pcm[10]).toBe(32767); // +2.0 clamps
  });
});
