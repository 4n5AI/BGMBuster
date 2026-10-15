/// <reference lib="webworker" />
/**
 * Separation worker: downloads and loads the model, then separates channel by channel in overlapping chunks.
 * Cancellation is done by terminating the worker from the main thread.
 */
import { MODEL } from '../config.ts';
import { gainToTarget, integratedLoudness } from '../audio/loudness.ts';
import { ChunkAssembler, planChunks } from './chunking.ts';
import { loadModelBytes, ModelError } from './modelCache.ts';
import { MrxModel } from './mrx.ts';
import type { SourceName } from './model.ts';
import type { ErrorKind, ProgressMessage, ResultMessage, WorkerRequest, WorkerResponse } from './protocol.ts';

declare const self: DedicatedWorkerGlobalScope;

let model: MrxModel | null = null;

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

function classify(e: unknown): { kind: ErrorKind; message: string; detail?: string } {
  if (e instanceof ModelError) return { kind: e.kind === 'download' ? 'model-download' : 'model-corrupt', message: e.message };
  const message = e instanceof Error ? e.message : String(e);
  const detail = e instanceof Error && e.stack ? e.stack : undefined;
  if (e instanceof RangeError || /out of memory|memory access out of bounds|allocation failed|Cannot enlarge memory|OOM/i.test(message)) {
    return { kind: 'memory', message, detail };
  }
  if (/WebAssembly|not supported|unsupported|no available backend/i.test(message)) return { kind: 'unsupported', message, detail };
  return { kind: 'unknown', message, detail };
}

async function handleLoad(req: Extract<WorkerRequest, { type: 'load' }>): Promise<void> {
  const t0 = performance.now();
  const bytes = await loadModelBytes(req.modelUrl, MODEL.cacheName, { sha256: req.sha256, bytes: req.bytes }, (loaded, total) => {
    post({ type: 'progress', stage: 'download', fraction: total > 0 ? loaded / total : 0, loaded, total });
  });
  post({ type: 'progress', stage: 'load', fraction: 0 });
  if (model) await model.dispose();
  model = new MrxModel();
  const backend = await model.load(bytes, req.prefer);
  const info = model.info;
  post({
    type: 'loaded',
    backend,
    elapsedSeconds: (performance.now() - t0) / 1000,
    adapter: info?.adapter,
    reason: info?.reason,
    selfTest: info?.selfTest ?? 'skipped',
  });
}

async function handleSeparate(req: Extract<WorkerRequest, { type: 'separate' }>): Promise<void> {
  if (!model) throw new Error('model not loaded');
  const sr = model.sampleRate;
  const { channels } = req;
  const length = channels[0].length;
  const loudness = integratedLoudness(channels, sr);
  const gain = gainToTarget(loudness, req.targetLufs);
  const chunkLength = Math.max(1, Math.round(req.chunkSeconds * sr));
  const overlap = Math.min(chunkLength - 1, Math.max(0, Math.round(req.overlapSeconds * sr)));
  const ranges = planChunks(length, chunkLength, overlap);
  const total = ranges.length * channels.length;
  const t0 = performance.now();
  let done = 0;
  const result: Record<SourceName, Float32Array[]> = { music: [], speech: [], sfx: [] };
  const scaled = new Float32Array(chunkLength);
  for (let c = 0; c < channels.length; c++) {
    const ch = channels[c];
    const assemblers = {
      music: new ChunkAssembler(length, ranges, req.crossfade),
      speech: new ChunkAssembler(length, ranges, req.crossfade),
      sfx: new ChunkAssembler(length, ranges, req.crossfade),
    };
    for (let k = 0; k < ranges.length; k++) {
      const { start, end } = ranges[k];
      const n = end - start;
      const input = scaled.subarray(0, n);
      for (let i = 0; i < n; i++) input[i] = ch[start + i] * gain;
      const stems = await model.separateChunk(input);
      for (const name of model.sources) assemblers[name].add(k, stems[name]);
      done++;
      const elapsed = (performance.now() - t0) / 1000;
      const progress: ProgressMessage = {
        type: 'progress',
        stage: 'separate',
        fraction: done / total,
        etaSeconds: (elapsed / done) * (total - done),
      };
      post(progress);
    }
    const inv = 1 / gain;
    for (const name of model.sources) {
      const y = assemblers[name].finish();
      if (inv !== 1) for (let i = 0; i < y.length; i++) y[i] *= inv;
      result[name].push(y);
    }
  }
  const message: ResultMessage = {
    type: 'result',
    music: result.music,
    speech: result.speech,
    sfx: result.sfx,
    loudness,
    gain,
    elapsedSeconds: (performance.now() - t0) / 1000,
    chunks: ranges.length,
  };
  post(message, [...result.music, ...result.speech, ...result.sfx].map((a) => a.buffer));
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'load') await handleLoad(req);
    else if (req.type === 'separate') await handleSeparate(req);
  } catch (e) {
    const c = classify(e);
    post({ type: 'error', ...c });
  }
};
