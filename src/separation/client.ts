/** Main-thread handle on the separation worker: promise API, progress callbacks, and cancellation. */
import type { Crossfade } from './chunking.ts';
import type { Backend, BackendPreference } from './model.ts';
import type { ErrorKind, ProgressMessage, ResultMessage, WorkerRequest, WorkerResponse } from './protocol.ts';

export class SeparationError extends Error {
  readonly kind: ErrorKind;
  readonly detail: string | undefined;

  constructor(kind: ErrorKind, message: string, detail?: string) {
    super(message);
    this.name = 'SeparationError';
    this.kind = kind;
    this.detail = detail;
  }
}

export class CanceledError extends Error {
  constructor() {
    super('canceled');
    this.name = 'CanceledError';
  }
}

export interface LoadOptions {
  modelUrl: string;
  sha256: string;
  bytes: number;
  prefer: BackendPreference;
}

export interface SeparateOptions {
  chunkSeconds: number;
  overlapSeconds: number;
  crossfade: Crossfade;
  targetLufs: number;
}

export type ProgressHandler = (p: ProgressMessage) => void;

export interface LoadResult {
  backend: Backend;
  elapsedSeconds: number;
  adapter?: string;
  reason?: string;
  selfTest: 'passed' | 'failed' | 'skipped';
}

export type SeparationResult = Omit<ResultMessage, 'type'>;

interface Pending {
  resolve: (value: WorkerResponse) => void;
  reject: (error: Error) => void;
  onProgress?: ProgressHandler;
}

export function createSeparationWorker(): Worker {
  return new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
}

export class SeparationClient {
  private worker: Worker | null = null;
  private pending: Pending | null = null;
  private readonly createWorker: () => Worker;
  backend: Backend | null = null;

  constructor(createWorker: () => Worker = createSeparationWorker) {
    this.createWorker = createWorker;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const w = this.createWorker();
    w.onmessage = (event: MessageEvent<WorkerResponse>) => this.onMessage(event.data);
    w.onerror = (event: ErrorEvent) => {
      this.pending?.reject(new SeparationError('unknown', event.message || 'worker error'));
      this.pending = null;
    };
    this.worker = w;
    return w;
  }

  private onMessage(message: WorkerResponse): void {
    const pending = this.pending;
    if (!pending) return;
    if (message.type === 'progress') {
      pending.onProgress?.(message);
      return;
    }
    this.pending = null;
    if (message.type === 'error') pending.reject(new SeparationError(message.kind, message.message, message.detail));
    else pending.resolve(message);
  }

  private request(req: WorkerRequest, onProgress?: ProgressHandler, transfer: Transferable[] = []): Promise<WorkerResponse> {
    if (this.pending) return Promise.reject(new SeparationError('unknown', 'another operation is in progress'));
    const worker = this.ensureWorker();
    return new Promise<WorkerResponse>((resolve, reject) => {
      this.pending = { resolve, reject, onProgress };
      worker.postMessage(req, transfer);
    });
  }

  /** Downloads (once) and loads the model; resolves to the backend in use. */
  async load(options: LoadOptions, onProgress?: ProgressHandler): Promise<LoadResult> {
    const res = await this.request({ type: 'load', ...options }, onProgress);
    if (res.type !== 'loaded') throw new SeparationError('unknown', `unexpected response ${res.type}`);
    this.backend = res.backend;
    return { backend: res.backend, elapsedSeconds: res.elapsedSeconds, adapter: res.adapter, reason: res.reason, selfTest: res.selfTest };
  }

  /**
   * Separates the given channels (model sample rate). The arrays are copied to the worker, so the caller keeps
   * its own data. Cancel with `cancel()`.
   */
  async separate(channels: Float32Array[], options: SeparateOptions, onProgress?: ProgressHandler): Promise<SeparationResult> {
    const res = await this.request({ type: 'separate', channels, ...options }, onProgress);
    if (res.type !== 'result') throw new SeparationError('unknown', `unexpected response ${res.type}`);
    const { type: _type, ...rest } = res;
    return rest;
  }

  /** Stops whatever is running; the next call starts a fresh worker (the model is reloaded from the cache). */
  cancel(): void {
    this.worker?.terminate();
    this.worker = null;
    this.backend = null;
    const pending = this.pending;
    this.pending = null;
    pending?.reject(new CanceledError());
  }

  get busy(): boolean {
    return this.pending !== null;
  }
}
