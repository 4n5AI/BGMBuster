/** Orchestration: file -> decode -> separate -> preview buffers, and the three ways of saving. */
import { CHUNK, EXPORT, INPUT_TARGET_LUFS, LIMITS, MODEL } from '../config.ts';
import { applyGain, clipSafeGain, remixChannel } from '../audio/remix.ts';
import { resampleChannels, toAudioBuffer } from '../audio/resample.ts';
import { encodeWav16 } from '../audio/wav.ts';
import { ExportError, exportVideo } from '../export/mux.ts';
import { chooseSaveTarget, replaceExtension } from '../export/save.ts';
import { formatDurationWords } from '../i18n/index.ts';
import type { MessageKey } from '../i18n/ja.ts';
import { decodeAudioTrack } from '../media/decode.ts';
import { MediaError, openMedia, type OpenedMedia } from '../media/probe.ts';
import { CanceledError, SeparationClient, SeparationError } from '../separation/client.ts';
import type { BackendPreference } from '../separation/model.ts';
import type { AppError, SeparationOutput, Store } from './state.ts';

void EXPORT;

export type StemName = 'speech' | 'sfx' | 'music';

export function browserSupported(): boolean {
  return (
    typeof WebAssembly !== 'undefined' &&
    typeof Worker !== 'undefined' &&
    typeof AudioContext !== 'undefined' &&
    typeof OfflineAudioContext !== 'undefined' &&
    typeof AudioDecoder !== 'undefined' &&
    typeof crypto !== 'undefined' &&
    !!crypto.subtle
  );
}

function maxDurationWords(): string {
  return formatDurationWords(LIMITS.maxDurationSeconds);
}

export function toAppError(e: unknown): AppError {
  const detail = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  if (e instanceof MediaError) {
    switch (e.kind) {
      case 'unsupported-format':
        return { key: 'error.format', actions: ['chooseFile'], detail };
      case 'no-audio':
        return { key: 'error.noAudio', actions: ['chooseFile'], detail };
      case 'too-long':
        return { key: 'error.tooLong', params: { max: maxDurationWords() }, actions: ['chooseFile'], detail };
      case 'undecodable-audio':
        return { key: 'error.audioCodec', actions: ['chooseFile'], detail };
    }
  }
  if (e instanceof SeparationError) {
    switch (e.kind) {
      case 'model-download':
        return { key: 'error.modelDownload', actions: ['retry'], detail: e.detail ?? detail };
      case 'model-corrupt':
        return { key: 'error.modelCorrupt', actions: ['reload'], detail: e.detail ?? detail };
      case 'memory':
        return { key: 'error.memory', actions: ['retry', 'chooseFile'], detail: e.detail ?? detail };
      case 'unsupported':
        return { key: 'error.browser', actions: [], detail: e.detail ?? detail };
      default:
        return { key: 'error.unknown', actions: ['retry'], detail: e.detail ?? detail };
    }
  }
  if (e instanceof ExportError) {
    if (e.kind === 'video-copy') return { key: 'error.videoCopy', actions: ['saveAudio'], detail };
    if (e.kind === 'no-encoder') return { key: 'error.browser', actions: ['saveAudio'], detail };
    return { key: 'error.unknown', actions: ['retry'], detail };
  }
  if (e instanceof RangeError || /out of memory|allocation failed|Array buffer allocation/i.test(detail)) {
    return { key: 'error.memory', actions: ['retry', 'chooseFile'], detail };
  }
  return { key: 'error.unknown', actions: ['retry'], detail };
}

/** `?backend=wasm` or `?backend=webgpu` forces a backend (diagnostics). */
function preferredBackend(): BackendPreference {
  const v = new URLSearchParams(location.search).get('backend');
  return v === 'wasm' || v === 'webgpu' ? v : 'auto';
}

function chunkSeconds(): number {
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return mem !== undefined && mem <= 4 ? 10 : CHUNK.seconds;
}

export class Pipeline {
  private readonly store: Store;
  private client = new SeparationClient();
  private media: OpenedMedia | null = null;
  private runId = 0;

  constructor(store: Store) {
    this.store = store;
  }

  private modelUrl(): string {
    return new URL(import.meta.env.BASE_URL + MODEL.file, location.href).href;
  }

  /** Steps 1: inspect the file. */
  async selectFile(file: File): Promise<void> {
    this.reset();
    const fileUrl = URL.createObjectURL(file);
    this.store.set({ file, fileUrl, phase: 'loaded', error: null, notes: [], canceled: false });
    try {
      const media = await openMedia(file, LIMITS.maxDurationSeconds);
      this.media = media;
      const notes: MessageKey[] = [];
      if (media.info.audioTrackCount > 1) notes.push('note.tracks');
      if (media.info.audio.channels > 2) notes.push('note.channels');
      this.store.set({ info: media.info, notes });
    } catch (e) {
      this.store.set({ phase: 'error', error: toAppError(e) });
    }
  }

  /** Step 2: model, decode, separate. */
  async run(): Promise<void> {
    const media = this.media;
    if (!media) return;
    const id = ++this.runId;
    const tech = { ...this.store.get().tech };
    try {
      this.store.set({ phase: 'preparing', canceled: false, error: null, progress: { stage: 'model', fraction: null } });
      const t0 = performance.now();
      const load = await this.client.load({ modelUrl: this.modelUrl(), sha256: MODEL.sha256, bytes: MODEL.bytes, prefer: preferredBackend() }, (p) => {
        if (id !== this.runId) return;
        if (p.stage === 'download') this.store.set({ progress: { stage: 'model', fraction: p.fraction, loaded: p.loaded, total: p.total } });
        else this.store.set({ progress: { stage: 'model', fraction: null } });
      });
      if (id !== this.runId) return;
      tech.backend = load.backend;
      tech.adapter = load.adapter;
      tech.backendReason = load.reason;
      tech.selfTest = load.selfTest;
      tech.modelLoadSeconds = (performance.now() - t0) / 1000;
      tech.wasmSlow = load.backend === 'wasm';
      this.store.set({ phase: 'processing', tech, progress: { stage: 'read', fraction: 0 } });

      const t1 = performance.now();
      const decoded = await decodeAudioTrack(media.audioTrack, (seconds, total) => {
        if (id === this.runId) this.store.set({ progress: { stage: 'read', fraction: total > 0 ? seconds / total : null } });
      });
      if (id !== this.runId) return;
      const modelRate = MODEL.sampleRate;
      const mix = await resampleChannels(decoded.channels, decoded.sampleRate, modelRate, 2);
      tech.decodeSeconds = (performance.now() - t1) / 1000;
      this.store.set({ tech, progress: { stage: 'separate', fraction: 0 } });

      const t2 = performance.now();
      const result = await this.client.separate(
        mix,
        { chunkSeconds: chunkSeconds(), overlapSeconds: CHUNK.overlapSeconds, crossfade: CHUNK.crossfade, targetLufs: INPUT_TARGET_LUFS },
        (p) => {
          if (id === this.runId && p.stage === 'separate') this.store.set({ progress: { stage: 'separate', fraction: p.fraction, etaSeconds: p.etaSeconds } });
        },
      );
      if (id !== this.runId) return;
      tech.separateSeconds = (performance.now() - t2) / 1000;
      const output: SeparationOutput = {
        sampleRate: modelRate,
        mix: toAudioBuffer(mix, modelRate),
        speech: toAudioBuffer(result.speech, modelRate),
        sfx: toAudioBuffer(result.sfx, modelRate),
        music: toAudioBuffer(result.music, modelRate),
        nativeSampleRate: decoded.sampleRate,
        audioOffsetSeconds: decoded.firstTimestamp,
        elapsedSeconds: result.elapsedSeconds,
        chunks: result.chunks,
      };
      this.store.set({ phase: 'done', output, progress: null, tech });
    } catch (e) {
      if (id !== this.runId) return;
      if (e instanceof CanceledError || this.store.get().canceled) {
        this.store.set({ phase: 'loaded', progress: null, canceled: true });
        return;
      }
      this.store.set({ phase: 'error', progress: null, error: toAppError(e) });
    }
  }

  cancel(): void {
    const state = this.store.get();
    if (state.phase !== 'preparing' && state.phase !== 'processing') return;
    this.runId++;
    this.store.set({ canceled: true, phase: 'loaded', progress: null });
    this.client.cancel();
    // Decoding is stopped by disposing the input; reopen it so the file can be processed again.
    const file = state.file;
    this.media?.input.dispose();
    this.media = null;
    if (file) {
      void openMedia(file, LIMITS.maxDurationSeconds)
        .then((m) => {
          this.media = m;
        })
        .catch(() => undefined);
    }
  }

  /** The re-mixed signal at the model rate, then at the file's own rate, with a clip-safe gain. */
  private async renderMix(): Promise<{ channels: Float32Array[]; sampleRate: number }> {
    const { output, method, musicLevel, sfxLevel } = this.store.get();
    if (!output) throw new Error('nothing to save');
    const channels = Array.from({ length: output.mix.numberOfChannels }, (_, c) =>
      remixChannel(
        method,
        { mix: output.mix.getChannelData(c), speech: output.speech.getChannelData(c), sfx: output.sfx.getChannelData(c), music: output.music.getChannelData(c) },
        musicLevel,
        sfxLevel,
      ),
    );
    applyGain(channels, clipSafeGain(channels));
    const native = await resampleChannels(channels, output.sampleRate, output.nativeSampleRate, 2);
    return { channels: native, sampleRate: output.nativeSampleRate };
  }

  async saveVideo(): Promise<void> {
    const state = this.store.get();
    const media = this.media;
    if (!media?.videoTrack || !state.file || state.saving) return;
    const t0 = performance.now();
    this.store.set({ saving: 'video', saveProgress: 0, error: null });
    try {
      const plan = media.info.container === 'webm' || media.info.container === 'mkv' ? { mime: 'video/webm', ext: '.webm' } : { mime: 'video/mp4', ext: '.mp4' };
      const target = await chooseSaveTarget(replaceExtension(state.file.name, '-nobgm', plan.ext), plan.mime, plan.ext);
      if (!target) return;
      try {
        const { channels, sampleRate } = await this.renderMix();
        const out = await exportVideo({
          input: media.input,
          videoTrack: media.videoTrack,
          container: media.info.container,
          audio: channels,
          sampleRate,
          audioOffsetSeconds: this.store.get().output?.audioOffsetSeconds ?? 0,
          originalBitrate: media.info.audio.bitrate,
          writable: target.writable,
          onProgress: (f) => this.store.set({ saveProgress: f }),
        });
        if (out.bytes) await target.finish(out.bytes, out.mimeType);
        this.store.set({ tech: { ...this.store.get().tech, exportSeconds: (performance.now() - t0) / 1000, exportCodec: out.audioCodec, exportBitrate: out.audioBitrate, encoderDelay: out.encoderDelay } });
      } catch (e) {
        await target.cancel();
        throw e;
      }
    } catch (e) {
      this.store.set({ error: toAppError(e) });
    } finally {
      this.store.set({ saving: null, saveProgress: null });
    }
  }

  async saveAudio(): Promise<void> {
    const state = this.store.get();
    if (!state.file || state.saving) return;
    this.store.set({ saving: 'audio', saveProgress: null, error: null });
    try {
      const target = await chooseSaveTarget(replaceExtension(state.file.name, '-nobgm', '.wav'), 'audio/wav', '.wav');
      if (!target) return;
      const { channels, sampleRate } = await this.renderMix();
      await this.writeWav(target, channels, sampleRate);
    } catch (e) {
      this.store.set({ error: toAppError(e) });
    } finally {
      this.store.set({ saving: null, saveProgress: null });
    }
  }

  async saveStem(stem: StemName): Promise<void> {
    const state = this.store.get();
    const output = state.output;
    if (!state.file || !output || state.saving) return;
    this.store.set({ saving: 'stem', saveProgress: null, error: null });
    try {
      const suffix = { speech: '-dialogue', sfx: '-effects', music: '-music' }[stem];
      const target = await chooseSaveTarget(replaceExtension(state.file.name, suffix, '.wav'), 'audio/wav', '.wav');
      if (!target) return;
      const buffer = output[stem];
      const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c).slice());
      applyGain(channels, clipSafeGain(channels));
      const native = await resampleChannels(channels, output.sampleRate, output.nativeSampleRate, 2);
      await this.writeWav(target, native, output.nativeSampleRate);
    } catch (e) {
      this.store.set({ error: toAppError(e) });
    } finally {
      this.store.set({ saving: null, saveProgress: null });
    }
  }

  private async writeWav(target: Awaited<ReturnType<typeof chooseSaveTarget>>, channels: Float32Array[], sampleRate: number): Promise<void> {
    if (!target) return;
    const wav = encodeWav16(channels, sampleRate);
    if (target.writable) {
      await target.writable.write(wav);
      await target.writable.close();
    } else {
      await target.finish(wav, 'audio/wav');
    }
  }

  reset(): void {
    this.runId++;
    this.client.cancel();
    this.media?.input.dispose();
    this.media = null;
    const { fileUrl } = this.store.get();
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    this.store.set({ phase: 'idle', file: null, fileUrl: null, info: null, notes: [], progress: null, output: null, error: null, saving: null, saveProgress: null, canceled: false, tech: {} });
  }

  hasVideo(): boolean {
    return !!this.media?.videoTrack;
  }
}
