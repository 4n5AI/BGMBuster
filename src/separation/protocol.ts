/** Messages between the main thread (client.ts) and the separation worker (worker.ts). */
import type { Crossfade } from './chunking.ts';
import type { Backend, BackendPreference } from './model.ts';

export type ErrorKind = 'model-download' | 'model-corrupt' | 'memory' | 'unsupported' | 'unknown';

export type WorkerRequest =
  | { type: 'load'; modelUrl: string; sha256: string; bytes: number; prefer: BackendPreference }
  | {
      type: 'separate';
      /** One Float32Array per channel at the model's sample rate. */
      channels: Float32Array[];
      chunkSeconds: number;
      overlapSeconds: number;
      crossfade: Crossfade;
      targetLufs: number;
    };

export interface ProgressMessage {
  type: 'progress';
  stage: 'download' | 'load' | 'separate';
  /** 0..1 within the stage. */
  fraction: number;
  loaded?: number;
  total?: number;
  etaSeconds?: number;
}

export interface ResultMessage {
  type: 'result';
  music: Float32Array[];
  speech: Float32Array[];
  sfx: Float32Array[];
  /** Integrated loudness of the input (LUFS) and the gain that was applied before inference. */
  loudness: number;
  gain: number;
  /** Wall-clock seconds for the separation and the number of chunks processed. */
  elapsedSeconds: number;
  chunks: number;
}

export type WorkerResponse =
  | ProgressMessage
  | { type: 'loaded'; backend: Backend; elapsedSeconds: number }
  | ResultMessage
  | { type: 'error'; kind: ErrorKind; message: string; detail?: string };
