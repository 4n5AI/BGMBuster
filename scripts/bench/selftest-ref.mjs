// Computes the backend self-test reference values (src/config.ts SELFTEST.reference) with the WebAssembly
// backend in Node, which matches the upstream PyTorch implementation. Re-run after changing the model file.
// usage: node --experimental-strip-types scripts/bench/selftest-ref.mjs [model path]
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { magnitudeFT, stft } from '../../src/audio/stft.ts';
import { maskStats, probeSignal } from '../../src/separation/selftest.ts';

const MRX_WINDOWS = [1024, 2048, 8192]; // same as src/separation/mrx.ts (not imported: its ?url imports are Vite-only)
const MRX_HOP = 256;
const modelPath = process.argv[2] ?? 'public/models/mrx-v1.int8.onnx';
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(modelPath)), { executionProviders: ['wasm'] });
const x = probeSignal();
const feeds = {};
const specs = [];
for (const n of MRX_WINDOWS) {
  const s = stft(x, n, MRX_HOP);
  specs.push(s);
  feeds[`mag_${n}`] = new ort.Tensor('float32', magnitudeFT(s), [1, s.bins, s.frames]);
}
const out = await session.run(feeds);
console.log(JSON.stringify(maskStats(out, specs)));
await session.release();
