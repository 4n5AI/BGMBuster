/**
 * Writing the result: the input's video track copied packet by packet (no re-encoding), the original
 * audio discarded, and a new audio track encoded from the re-mixed samples.
 */
import {
  AudioSample,
  AudioSampleSource,
  BufferTarget,
  canEncodeAudio,
  Conversion,
  Mp4OutputFormat,
  Output,
  Quality,
  StreamTarget,
  WebMOutputFormat,
  type AudioCodec,
  type Input,
  type InputVideoTrack,
  type OutputFormat,
} from 'mediabunny';
import { EXPORT } from '../config.ts';
import type { ContainerKind } from '../media/probe.ts';
import { measureEncoderDelay } from './encoderDelay.ts';

export class ExportError extends Error {
  readonly kind: 'video-copy' | 'no-encoder' | 'canceled';

  constructor(kind: 'video-copy' | 'no-encoder' | 'canceled', message: string) {
    super(message);
    this.name = 'ExportError';
    this.kind = kind;
  }
}

export interface ExportPlan {
  format: OutputFormat;
  codec: AudioCodec;
  extension: string;
  mimeType: string;
}

/** MP4 and MOV become MP4 with AAC; WebM and MKV become WebM with Opus. */
export function planExport(container: ContainerKind): ExportPlan {
  if (container === 'webm' || container === 'mkv') {
    return { format: new WebMOutputFormat(), codec: 'opus', extension: '.webm', mimeType: 'video/webm' };
  }
  return { format: new Mp4OutputFormat({ fastStart: 'in-memory' }), codec: 'aac', extension: '.mp4', mimeType: 'video/mp4' };
}

let aacRegistered = false;

/** Makes sure the chosen audio codec can be encoded, registering the AAC polyfill when the browser lacks it. */
export async function ensureAudioEncoder(codec: AudioCodec, channels: number, sampleRate: number, bitrate: number): Promise<void> {
  const config = { numberOfChannels: channels, sampleRate, quality: new Quality({ bitrate }) };
  if (await canEncodeAudio(codec, config)) return;
  if (codec === 'aac' && !aacRegistered) {
    const { registerAacEncoder } = await import('@mediabunny/aac-encoder');
    registerAacEncoder();
    aacRegistered = true;
    if (await canEncodeAudio(codec, config)) return;
  }
  throw new ExportError('no-encoder', `this browser cannot encode ${codec}`);
}

export interface ExportVideoOptions {
  input: Input;
  videoTrack: InputVideoTrack;
  container: ContainerKind;
  /** The new audio, at `sampleRate`, starting at `audioOffsetSeconds` on the output timeline. */
  audio: Float32Array[];
  sampleRate: number;
  audioOffsetSeconds: number;
  /** Bitrate of the original audio, if known; the output uses at least EXPORT.aacMinBitrate. */
  originalBitrate: number | null;
  /** When given, the file is streamed here (File System Access API); otherwise it is built in memory. */
  writable?: WritableStream<{ data: Uint8Array; position: number }> | FileSystemWritableFileStream;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

export interface ExportVideoResult {
  /** The file bytes when no writable was given. */
  bytes: ArrayBuffer | null;
  mimeType: string;
  extension: string;
  audioCodec: AudioCodec;
  audioBitrate: number;
  /** Encoder delay (samples) that was compensated; null when it could not be measured. */
  encoderDelay: number | null;
}

function pickBitrate(codec: AudioCodec, originalBitrate: number | null): number {
  const floor = codec === 'aac' ? EXPORT.aacMinBitrate : EXPORT.opusBitrate;
  if (!originalBitrate || !Number.isFinite(originalBitrate)) return floor;
  return Math.min(320_000, Math.max(floor, Math.round(originalBitrate / 1000) * 1000));
}

/**
 * Copies the video track and muxes the new audio. Throws ExportError('video-copy') when the video cannot be
 * copied into the output container (never re-encodes the picture).
 */
export async function exportVideo(options: ExportVideoOptions): Promise<ExportVideoResult> {
  const plan = planExport(options.container);
  const channels = options.audio.length;
  const bitrate = pickBitrate(plan.codec, options.originalBitrate);
  await ensureAudioEncoder(plan.codec, channels, options.sampleRate, bitrate);
  // Delay of encode -> mux -> decode for this codec on this browser (Chrome's native AAC encoders prime with
  // 2112 samples, the polyfill with 1024); 1024 when it cannot be measured, Opus in WebM carries its pre-skip itself.
  const measured = await measureEncoderDelay({ codec: plan.codec, sampleRate: options.sampleRate, channels, bitrate }).catch(() => null);
  const delaySamples = measured ?? (plan.codec === 'aac' ? 1024 : 0);
  const target = options.writable ? new StreamTarget(options.writable as WritableStream<{ data: Uint8Array; position: number }>) : new BufferTarget();
  const output = new Output({ format: plan.format, target });
  const conversion = await Conversion.init({
    input: options.input,
    output,
    audio: { discard: true },
    video: (track) => (track.id === options.videoTrack.id ? {} : { discard: true }),
    copy: { mode: 'forced' },
    composable: true,
  });
  const copied = conversion.utilizedTracks.some((t) => t.id === options.videoTrack.id);
  if (!conversion.isValid || !copied) {
    const reason = conversion.discardedTracks.find((d) => d.track.id === options.videoTrack.id)?.reason ?? 'unknown';
    await conversion.cancel();
    throw new ExportError('video-copy', `video track cannot be copied into ${plan.extension} (${reason})`);
  }
  const audioSource = new AudioSampleSource({ codec: plan.codec, quality: new Quality({ bitrate }) });
  output.addAudioTrack(audioSource);
  try {
    const tags = await options.input.getMetadataTags();
    output.setMetadataTags(tags);
  } catch {
    // tags are optional
  }
  await output.start();
  const length = options.audio[0].length;
  const sr = options.sampleRate;
  const sliceFrames = sr * 10;
  // Samples are placed on the timeline at audioOffsetSeconds - delay. Negative timestamps are expressed by
  // the muxer as an edit list, which players use to skip the encoder priming.
  const startSeconds = options.audioOffsetSeconds - delaySamples / sr;
  const addSlice = async (slice: Float32Array[], timestamp: number): Promise<void> => {
    const frames = slice[0].length;
    const data = new Float32Array(frames * channels);
    slice.forEach((ch, c) => data.set(ch, c * frames));
    const sample = new AudioSample({ data, format: 'f32-planar', numberOfChannels: channels, sampleRate: sr, timestamp });
    try {
      await audioSource.add(sample);
    } finally {
      sample.close();
    }
  };
  try {
    let pos = 0;
    while (pos < length) {
      if (options.signal?.aborted) throw new ExportError('canceled', 'canceled');
      const end = Math.min(length, pos + sliceFrames);
      await addSlice(options.audio.map((ch) => ch.subarray(pos, end)), startSeconds + pos / sr);
      pos = end;
      await conversion.execute({ until: Math.max(0, startSeconds + pos / sr) });
      options.onProgress?.(pos / length);
    }
    audioSource.close();
    await conversion.execute();
    await output.finalize();
  } catch (e) {
    await conversion.cancel().catch(() => undefined);
    await output.cancel().catch(() => undefined);
    throw e;
  }
  return {
    bytes: target instanceof BufferTarget ? target.buffer : null,
    mimeType: plan.mimeType,
    extension: plan.extension,
    audioCodec: plan.codec,
    audioBitrate: bitrate,
    encoderDelay: measured,
  };
}
