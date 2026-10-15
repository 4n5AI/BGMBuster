// Benchmark the MRX ONNX graph with onnxruntime-web's WASM backend under Node (single thread, like a
// GitHub Pages site without cross-origin isolation). Measures session creation and one run per chunk length.
//
// usage: node scripts/bench/ort-web-node.mjs <model.onnx> [seconds=20] [batch=2] [runs=2]
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';

const [modelPath, secondsArg = '20', batchArg = '2', runsArg = '2'] = process.argv.slice(2);
const seconds = Number(secondsArg);
const batch = Number(batchArg);
const runs = Number(runsArg);
const HOP = 256;
const SR = 44100;
const frames = 1 + Math.floor((seconds * SR) / HOP);

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'warning';

const bytes = readFileSync(modelPath);
const t0 = performance.now();
const session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ['wasm'] });
const tCreate = performance.now() - t0;
console.log(`${modelPath}: ${(bytes.length / 1e6).toFixed(1)} MB, session created in ${(tCreate / 1000).toFixed(2)} s`);
console.log(`inputs: ${session.inputNames.join(', ')} | outputs: ${session.outputNames.join(', ')}`);

const feeds = {};
for (const n of [1024, 2048, 8192]) {
  const bins = n / 2 + 1;
  const data = new Float32Array(batch * bins * frames);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 0.05;
  feeds[`mag_${n}`] = new ort.Tensor('float32', data, [batch, bins, frames]);
}
for (let r = 0; r < runs; r++) {
  const t1 = performance.now();
  const out = await session.run(feeds);
  const dt = (performance.now() - t1) / 1000;
  const m = out.mask_1024;
  console.log(`run ${r + 1}: ${seconds} s x ${batch} ch (T=${frames}) -> ${dt.toFixed(2)} s  (${(dt / (seconds * batch)).toFixed(2)} s per audio-second-channel; mask_1024 dims ${m.dims.join('x')}, mean ${(m.data.reduce((a, b) => a + b, 0) / m.data.length).toFixed(4)})`);
}
const mem = process.memoryUsage();
console.log(`rss ${(mem.rss / 1e6).toFixed(0)} MB, heap ${(mem.heapUsed / 1e6).toFixed(0)} MB`);
await session.release();
