/** Benchmark page: times model loading and separation on this device (see bench.html). */
import { CHUNK, INPUT_TARGET_LUFS, MODEL } from './config.ts';
import { CanceledError, SeparationClient, SeparationError } from './separation/client.ts';
import type { BackendPreference } from './separation/model.ts';

const logEl = document.getElementById('log') as HTMLPreElement;
const lines: string[] = [];
function log(line: string): void {
  lines.push(line);
  logEl.textContent = lines.join('\n');
  console.log(line);
}

interface BenchResult {
  backend: string;
  adapter?: string;
  seconds: number;
  channels: number;
  chunks: number;
  loadSeconds: number;
  separateSeconds: number;
  perAudioSecondChannel: number;
  jsHeapMB?: number;
  userAgent: string;
  hardwareConcurrency: number;
  error?: string;
}

declare global {
  interface Window {
    __benchResults: BenchResult[];
    __benchRun: (prefer: BackendPreference) => Promise<BenchResult>;
    __benchStems?: { music: Float32Array[]; speech: Float32Array[]; sfx: Float32Array[] };
  }
}

const params = new URLSearchParams(location.search);

function synthetic(seconds: number, sr: number): Float32Array[] {
  const n = Math.round(seconds * sr);
  const l = new Float32Array(n);
  const r = new Float32Array(n);
  let state = 12345;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    state = (state * 1664525 + 1013904223) >>> 0;
    const noise = (state / 4294967296 - 0.5) * 0.2 * (Math.sin(2 * Math.PI * 1.5 * t) > 0.8 ? 1 : 0);
    const music = 0.2 * (Math.sin(2 * Math.PI * 220 * t) + 0.5 * Math.sin(2 * Math.PI * 330 * t));
    const speech = 0.3 * Math.sin(2 * Math.PI * 150 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 4 * t));
    l[i] = music + speech + noise;
    r[i] = 0.8 * l[i];
  }
  return [l, r];
}

let client: SeparationClient | null = null;

async function run(prefer: BackendPreference): Promise<BenchResult> {
  const seconds = Number((document.getElementById('seconds') as HTMLInputElement).value) || 10;
  const chunkSeconds = Number((document.getElementById('chunk') as HTMLInputElement).value) || CHUNK.seconds;
  const overlapSeconds = Number((document.getElementById('overlap') as HTMLInputElement).value);
  client?.cancel();
  client = new SeparationClient();
  const modelUrl = new URL(import.meta.env.BASE_URL + MODEL.file, location.href).href;
  const result: BenchResult = {
    backend: 'n/a',
    seconds,
    channels: 2,
    chunks: 0,
    loadSeconds: 0,
    separateSeconds: 0,
    perAudioSecondChannel: 0,
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
  };
  log(`--- ${new Date().toISOString()} prefer=${prefer} ${seconds}s stereo, chunk ${chunkSeconds}s, overlap ${overlapSeconds}s`);
  log(`crossOriginIsolated=${String(crossOriginIsolated)} hardwareConcurrency=${navigator.hardwareConcurrency} webgpu=${navigator.gpu ? 'yes' : 'no'}`);
  try {
    const t0 = performance.now();
    let lastPct = -1;
    const load = await client.load({ modelUrl, sha256: MODEL.sha256, bytes: MODEL.bytes, prefer }, (p) => {
      const pct = Math.floor(p.fraction * 10) * 10;
      if (p.stage === 'download' && pct !== lastPct) {
        lastPct = pct;
        log(`download ${pct}% (${((p.loaded ?? 0) / 1e6).toFixed(1)} MB)`);
      }
    });
    result.loadSeconds = (performance.now() - t0) / 1000;
    result.backend = load.backend;
    log(`model loaded: backend=${load.backend} in ${result.loadSeconds.toFixed(2)} s`);
    const inputUrls = [params.get('inputL'), params.get('inputR')].filter((u): u is string => !!u);
    const channels = inputUrls.length
      ? await Promise.all(inputUrls.map(async (u) => new Float32Array(await (await fetch(u)).arrayBuffer())))
      : synthetic(seconds, MODEL.sampleRate);
    result.channels = channels.length;
    result.seconds = channels[0].length / MODEL.sampleRate;
    const t1 = performance.now();
    const sep = await client.separate(
      channels,
      { chunkSeconds, overlapSeconds, crossfade: CHUNK.crossfade, targetLufs: INPUT_TARGET_LUFS },
      (p) => {
        if (p.stage === 'separate') log(`separate ${(p.fraction * 100).toFixed(0)}% eta ${(p.etaSeconds ?? 0).toFixed(0)} s`);
      },
    );
    result.separateSeconds = (performance.now() - t1) / 1000;
    result.chunks = sep.chunks;
    result.perAudioSecondChannel = result.separateSeconds / (result.seconds * channels.length);
    window.__benchStems = { music: sep.music, speech: sep.speech, sfx: sep.sfx };
    const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    if (mem) result.jsHeapMB = mem.usedJSHeapSize / 1e6;
    const rms = (a: Float32Array): number => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
    log(`separated in ${result.separateSeconds.toFixed(2)} s (${sep.chunks} chunks x 2 ch, ${result.perAudioSecondChannel.toFixed(3)} s per audio-second-channel; worker ${sep.elapsedSeconds.toFixed(2)} s)`);
    log(`input ${sep.loudness.toFixed(1)} LUFS gain ${sep.gain.toFixed(3)}; rms music ${rms(sep.music[0]).toFixed(4)} speech ${rms(sep.speech[0]).toFixed(4)} sfx ${rms(sep.sfx[0]).toFixed(4)}` + (result.jsHeapMB ? `; JS heap ${result.jsHeapMB.toFixed(0)} MB` : ''));
  } catch (e) {
    if (e instanceof CanceledError) {
      log('canceled');
      result.error = 'canceled';
    } else {
      const msg = e instanceof SeparationError ? `${e.kind}: ${e.message}` : e instanceof Error ? e.message : String(e);
      log(`ERROR ${msg}`);
      result.error = msg;
    }
  }
  window.__benchResults.push(result);
  return result;
}

window.__benchResults = [];
window.__benchRun = run;
document.getElementById('run-auto')?.addEventListener('click', () => void run('auto'));
document.getElementById('run-webgpu')?.addEventListener('click', () => void run('webgpu'));
document.getElementById('run-wasm')?.addEventListener('click', () => void run('wasm'));
document.getElementById('cancel')?.addEventListener('click', () => client?.cancel());

for (const key of ['seconds', 'chunk', 'overlap'] as const) {
  const v = params.get(key);
  if (v !== null) (document.getElementById(key) as HTMLInputElement).value = v;
}
const auto = params.get('auto');
if (auto === 'auto' || auto === 'webgpu' || auto === 'wasm') void run(auto);
