/**
 * MRX (MERL, "The Cocktail Fork Problem", ICASSP 2022) running on onnxruntime-web.
 * The ONNX graph holds the network between the STFTs and the iSTFTs (see scripts/export/build_onnx.py);
 * the three-resolution STFT, masking and iSTFT happen here in TypeScript.
 */
import wasmCpuUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import wasmGpuUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { MODEL } from '../config.ts';
import { istft, magnitudeFT, stft, type Spectrogram } from '../audio/stft.ts';
import type { Backend, BackendPreference, SeparationModel, SourceName, Stems } from './model.ts';

type Ort = typeof import('onnxruntime-web');
type Session = import('onnxruntime-web').InferenceSession;

export const MRX_WINDOWS = [1024, 2048, 8192] as const;
export const MRX_HOP = 256;

export interface MrxLoadInfo {
  backend: Backend;
  /** Description of the GPU adapter when WebGPU is used. */
  adapter?: string;
  reason?: string;
}

async function gpuAvailable(): Promise<string | null> {
  const nav = (globalThis as { navigator?: Navigator | WorkerNavigator }).navigator;
  const gpu = nav?.gpu;
  if (!gpu) return null;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
    return [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(' ') || 'webgpu';
  } catch {
    return null;
  }
}

export class MrxModel implements SeparationModel {
  readonly id = MODEL.id;
  readonly license = MODEL.license;
  readonly sampleRate = MODEL.sampleRate;
  readonly sources = MODEL.sources;
  private ort: Ort | null = null;
  private session: Session | null = null;
  info: MrxLoadInfo | null = null;

  async load(modelBytes: ArrayBuffer, prefer: BackendPreference): Promise<Backend> {
    const bytes = new Uint8Array(modelBytes);
    if (prefer !== 'wasm') {
      const adapter = await gpuAvailable();
      if (adapter) {
        try {
          const ort = (await import('onnxruntime-web/webgpu')) as Ort;
          ort.env.wasm.wasmPaths = { wasm: wasmGpuUrl };
          const session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
          this.ort = ort;
          this.session = session;
          this.info = { backend: 'webgpu', adapter };
          return 'webgpu';
        } catch (e) {
          if (prefer === 'webgpu') throw e;
          this.info = { backend: 'wasm', reason: `WebGPU failed: ${e instanceof Error ? e.message : String(e)}` };
        }
      } else if (prefer === 'webgpu') {
        throw new Error('WebGPU is not available');
      } else {
        this.info = { backend: 'wasm', reason: 'no WebGPU adapter' };
      }
    }
    const ort = (await import('onnxruntime-web/wasm')) as Ort;
    ort.env.wasm.wasmPaths = { wasm: wasmCpuUrl };
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    this.ort = ort;
    this.session = session;
    this.info ??= { backend: 'wasm' };
    return 'wasm';
  }

  async separateChunk(mono: Float32Array): Promise<Stems> {
    if (!this.ort || !this.session) throw new Error('MrxModel: load() first');
    const feeds: Record<string, import('onnxruntime-web').Tensor> = {};
    const specs: Spectrogram[] = [];
    for (const n of MRX_WINDOWS) {
      const spec = stft(mono, n, MRX_HOP);
      specs.push(spec);
      feeds[`mag_${n}`] = new this.ort.Tensor('float32', magnitudeFT(spec), [1, spec.bins, spec.frames]);
    }
    const results = await this.session.run(feeds);
    const length = mono.length;
    const out: Stems = { music: new Float32Array(length), speech: new Float32Array(length), sfx: new Float32Array(length) };
    for (let r = 0; r < MRX_WINDOWS.length; r++) {
      const spec = specs[r];
      const tensor = results[`mask_${MRX_WINDOWS[r]}`];
      const data = tensor.data as Float32Array;
      const planeSize = spec.bins * spec.frames;
      for (let s = 0; s < this.sources.length; s++) {
        const name: SourceName = this.sources[s];
        istft(spec, data.subarray(s * planeSize, (s + 1) * planeSize), out[name]);
      }
      tensor.dispose?.();
    }
    return out;
  }

  async dispose(): Promise<void> {
    await this.session?.release();
    this.session = null;
    this.ort = null;
  }
}
