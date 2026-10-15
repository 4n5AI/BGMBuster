// Check src/audio/stft.ts against the NumPy vectors from make_stft_vectors.py (torch.stft conventions).
// usage: node --experimental-strip-types scripts/parity/check_stft.mjs <vector_dir>
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { stft, istft, magnitudeFT } from '../../src/audio/stft.ts';

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
const x = new Float32Array(readFileSync(join(dir, 'x.f32')).buffer.slice(0));
const snr = (est, ref) => {
  let s = 0, e = 0;
  for (let i = 0; i < ref.length; i++) { s += ref[i] * ref[i]; e += (est[i] - ref[i]) ** 2; }
  return 10 * Math.log10(s / (e + 1e-30));
};
let ok = true;
for (const [n, info] of Object.entries(meta.resolutions)) {
  const nFft = Number(n);
  const t0 = performance.now();
  const spec = stft(x, nFft, meta.hop);
  const tStft = performance.now() - t0;
  if (spec.bins !== info.bins || spec.frames !== info.frames) {
    console.log(`n_fft=${n}: shape mismatch ts ${spec.bins}x${spec.frames} vs numpy ${info.bins}x${info.frames}`);
    ok = false;
    continue;
  }
  const refInter = new Float32Array(readFileSync(join(dir, `spec_${n}.f32`)).buffer.slice(0));
  const refRe = new Float32Array(spec.frames * spec.bins);
  const refIm = new Float32Array(spec.frames * spec.bins);
  for (let i = 0; i < refRe.length; i++) { refRe[i] = refInter[2 * i]; refIm[i] = refInter[2 * i + 1]; }
  const sRe = snr(spec.re, refRe), sIm = snr(spec.im, refIm);
  const t1 = performance.now();
  const y = istft(spec);
  const tIstft = performance.now() - t1;
  const sRt = snr(y, x);
  const mag = magnitudeFT(spec);
  let magMax = 0; for (const v of mag) magMax = Math.max(magMax, v);
  console.log(`n_fft=${n}: STFT vs numpy re ${sRe.toFixed(1)} dB, im ${sIm.toFixed(1)} dB; iSTFT round trip ${sRt.toFixed(1)} dB; ` +
    `stft ${tStft.toFixed(0)} ms, istft ${tIstft.toFixed(0)} ms for ${(x.length / meta.sampleRate).toFixed(1)} s; max |mag| ${magMax.toFixed(3)}`);
  if (sRe < 100 || sIm < 100 || sRt < 100) ok = false;
}
// warm timing: a second pass per resolution, STFT once + iSTFT three times (one per source)
for (const n of Object.keys(meta.resolutions)) {
  const nFft = Number(n);
  const t0 = performance.now();
  const spec = stft(x, nFft, meta.hop);
  const t1 = performance.now();
  for (let s = 0; s < 3; s++) istft(spec);
  const t2 = performance.now();
  console.log(`warm n_fft=${n}: stft ${((t1 - t0) / (x.length / meta.sampleRate)).toFixed(0)} ms per audio-second, 3x istft ${((t2 - t1) / (x.length / meta.sampleRate)).toFixed(0)} ms per audio-second`);
}
console.log(ok ? 'OK' : 'FAILED');
process.exit(ok ? 0 : 1);
