/** Application state: one store, a few phases, and subscribers that re-render. */
import type { ListenMode } from '../audio/player.ts';
import type { RemixMethod } from '../audio/remix.ts';
import type { MessageKey } from '../i18n/ja.ts';
import type { MediaInfo } from '../media/probe.ts';
import type { Backend } from '../separation/model.ts';

export type Phase = 'idle' | 'loaded' | 'preparing' | 'processing' | 'done' | 'error';

export type ProgressStage = 'model' | 'read' | 'separate' | 'write';

export interface Progress {
  stage: ProgressStage;
  /** 0..1 within the stage; null for indeterminate. */
  fraction: number | null;
  etaSeconds?: number;
  loaded?: number;
  total?: number;
}

export type ErrorAction = 'retry' | 'reload' | 'chooseFile' | 'saveAudio';

export interface AppError {
  key: MessageKey;
  params?: Record<string, string | number>;
  actions: ErrorAction[];
  detail?: string;
}

export interface SeparationOutput {
  /** All at the model sample rate, one Float32Array per channel. */
  sampleRate: number;
  mix: Float32Array[];
  speech: Float32Array[];
  sfx: Float32Array[];
  music: Float32Array[];
  /** The input's own sample rate and audio start offset, for writing files. */
  nativeSampleRate: number;
  audioOffsetSeconds: number;
  elapsedSeconds: number;
  chunks: number;
}

export interface Tech {
  backend?: Backend;
  adapter?: string;
  modelLoadSeconds?: number;
  decodeSeconds?: number;
  separateSeconds?: number;
  exportSeconds?: number;
  encoderDelay?: number | null;
  wasmSlow?: boolean;
}

export interface AppState {
  phase: Phase;
  file: File | null;
  fileUrl: string | null;
  info: MediaInfo | null;
  notes: MessageKey[];
  progress: Progress | null;
  canceled: boolean;
  output: SeparationOutput | null;
  listen: ListenMode;
  musicLevel: number;
  method: RemixMethod;
  saving: 'video' | 'audio' | 'stem' | null;
  saveProgress: number | null;
  error: AppError | null;
  tech: Tech;
}

export const initialState: AppState = {
  phase: 'idle',
  file: null,
  fileUrl: null,
  info: null,
  notes: [],
  progress: null,
  canceled: false,
  output: null,
  listen: 'removed',
  musicLevel: 0,
  method: 'B',
  saving: null,
  saveProgress: null,
  error: null,
  tech: {},
};

export class Store {
  private state: AppState = { ...initialState };
  private readonly listeners = new Set<(state: AppState, prev: AppState) => void>();

  get(): AppState {
    return this.state;
  }

  set(patch: Partial<AppState>): void {
    const prev = this.state;
    this.state = { ...prev, ...patch };
    for (const fn of this.listeners) fn(this.state, prev);
  }

  subscribe(fn: (state: AppState, prev: AppState) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}
