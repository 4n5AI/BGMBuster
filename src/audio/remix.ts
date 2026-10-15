/**
 * Re-mixing the separated sounds with a music level g in [0, 1].
 *   method A: out = speech + sfx + g * music
 *   method B: out = mix - (1 - g) * music   (keeps everything the model did not call music)
 */
export type RemixMethod = 'A' | 'B';

export interface StemChannels {
  mix: Float32Array;
  speech: Float32Array;
  sfx: Float32Array;
  music: Float32Array;
}

export function remixChannel(method: RemixMethod, s: StemChannels, musicLevel: number, out?: Float32Array): Float32Array {
  const n = s.mix.length;
  const y = out ?? new Float32Array(n);
  const g = Math.min(1, Math.max(0, musicLevel));
  if (method === 'A') {
    for (let i = 0; i < n; i++) y[i] = s.speech[i] + s.sfx[i] + g * s.music[i];
  } else {
    const k = 1 - g;
    for (let i = 0; i < n; i++) y[i] = s.mix[i] - k * s.music[i];
  }
  return y;
}

/** Largest absolute sample value across channels. */
export function peak(channels: readonly Float32Array[]): number {
  let p = 0;
  for (const ch of channels) {
    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(ch[i]);
      if (a > p) p = a;
    }
  }
  return p;
}

/** Gain that keeps the peak within ±1.0: 1 when nothing clips, otherwise 1 / peak (no limiter). */
export function clipSafeGain(channels: readonly Float32Array[]): number {
  const p = peak(channels);
  return p > 1 ? 1 / p : 1;
}

export function applyGain(channels: Float32Array[], gain: number): void {
  if (gain === 1) return;
  for (const ch of channels) for (let i = 0; i < ch.length; i++) ch[i] *= gain;
}
