/** WAV (RIFF) encoding: 16-bit PCM, interleaved channels. */

export interface WavInfo {
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  frames: number;
}

export function encodeWav16(channels: readonly Float32Array[], sampleRate: number): ArrayBuffer {
  const nCh = channels.length;
  if (nCh === 0) throw new Error('encodeWav16: no channels');
  const frames = channels[0].length;
  const blockAlign = nCh * 2;
  const dataBytes = frames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, nCh, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  ascii(36, 'data');
  view.setUint32(40, dataBytes, true);
  const out = new Int16Array(buffer, 44, frames * nCh);
  for (let c = 0; c < nCh; c++) {
    const ch = channels[c];
    for (let i = 0, j = c; i < frames; i++, j += nCh) {
      const v = ch[i];
      const clamped = v >= 1 ? 1 : v <= -1 ? -1 : v;
      out[j] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
    }
  }
  return buffer;
}

/** Reads the header of a 16-bit PCM WAV written by encodeWav16 (used by tests and diagnostics). */
export function readWavInfo(buffer: ArrayBuffer): WavInfo {
  const view = new DataView(buffer);
  const tag = (offset: number): string => String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE' || tag(12) !== 'fmt ' || tag(36) !== 'data') throw new Error('not a simple PCM WAV');
  const channels = view.getUint16(22, true);
  const sampleRate = view.getUint32(24, true);
  const bitsPerSample = view.getUint16(34, true);
  const dataBytes = view.getUint32(40, true);
  return { channels, sampleRate, bitsPerSample, frames: dataBytes / (channels * (bitsPerSample / 8)) };
}
