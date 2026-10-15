/** The separation model abstraction: one implementation per model (MRX for v1). */

export type SourceName = 'music' | 'speech' | 'sfx';
export const SOURCE_NAMES: readonly SourceName[] = ['music', 'speech', 'sfx'];

export type Backend = 'webgpu' | 'wasm';
export type BackendPreference = Backend | 'auto';

export interface Stems {
  music: Float32Array;
  speech: Float32Array;
  sfx: Float32Array;
}

export interface SeparationModel {
  readonly id: string;
  readonly license: string;
  /** Sample rate the model expects. */
  readonly sampleRate: number;
  readonly sources: readonly SourceName[];
  /** Creates the inference session from the model bytes. Resolves to the backend in use. */
  load(modelBytes: ArrayBuffer, prefer: BackendPreference): Promise<Backend>;
  /** Separates one mono chunk (at `sampleRate`, already loudness-normalised) into the three sources. */
  separateChunk(mono: Float32Array): Promise<Stems>;
  dispose(): Promise<void>;
}
