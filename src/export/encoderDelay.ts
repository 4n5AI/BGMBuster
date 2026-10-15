/**
 * Audio encoders add priming samples (AAC-LC: 1024 with FFmpeg's encoder, 2112 with Apple's) and not every
 * encoder/muxer pair signals them, which would leave the audio late against the picture. This measures the
 * end-to-end delay of "encode -> mux -> demux -> decode" for the codec in use once per session, so the exporter
 * can shift its timestamps. The same decoder is used for the input, so whatever the decoder trims cancels out.
 */
import {
  AudioSample,
  AudioSampleSink,
  AudioSampleSource,
  BufferSource,
  BufferTarget,
  Input,
  MP4,
  Mp4OutputFormat,
  Output,
  Quality,
  WEBM,
  WebMOutputFormat,
  type AudioCodec,
} from 'mediabunny';

const cache = new Map<string, number>();

function probeSignal(length: number, sampleRate: number): Float32Array {
  // deterministic linear chirp (200 Hz -> 8 kHz) with soft edges: its cross-correlation peak is sharp, and
  // unlike noise it survives perceptual codecs (noise substitution would blur the waveform)
  const x = new Float32Array(length);
  const start = Math.floor(length * 0.2);
  const end = Math.floor(length * 0.8);
  const span = end - start;
  const taper = Math.min(Math.floor(span * 0.05), 512);
  const seconds = span / sampleRate;
  for (let i = start; i < end; i++) {
    const u = (i - start) / span;
    const phase = 2 * Math.PI * seconds * (200 * u + ((8000 - 200) * u * u) / 2);
    let env = 1;
    if (i - start < taper) env = 0.5 - 0.5 * Math.cos((Math.PI * (i - start)) / taper);
    else if (end - 1 - i < taper) env = 0.5 - 0.5 * Math.cos((Math.PI * (end - 1 - i)) / taper);
    x[i] = 0.5 * env * Math.sin(phase);
  }
  return x;
}

export interface LagResult {
  /** Lag of `b` relative to `a` in samples (positive: b is late). */
  lag: number;
  /** Normalised cross-correlation at the peak (1 = identical, ~0 = unrelated or silent). */
  similarity: number;
}

/** Cross-correlation peak of `b` against `a`, searched within ±maxLag. */
export function findLagDetailed(a: Float32Array, b: Float32Array, maxLag: number): LagResult {
  let bestLag = 0;
  let best = -Infinity;
  const n = Math.min(a.length, b.length);
  let ea = 0;
  let eb = 0;
  for (let i = 0; i < n; i++) {
    ea += a[i] * a[i];
    eb += b[i] * b[i];
  }
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let acc = 0;
    const from = Math.max(0, lag);
    const to = Math.min(n, n + lag);
    for (let i = from; i < to; i++) acc += a[i - lag] * b[i];
    if (acc > best) {
      best = acc;
      bestLag = lag;
    }
  }
  const norm = Math.sqrt(ea * eb);
  return { lag: bestLag, similarity: norm > 0 && Number.isFinite(best) ? best / norm : 0 };
}

/** Lag of `b` relative to `a` in samples (positive: b is late), searched within ±maxLag. */
export function findLag(a: Float32Array, b: Float32Array, maxLag: number): number {
  return findLagDetailed(a, b, maxLag).lag;
}

export interface EncoderDelayOptions {
  codec: AudioCodec;
  sampleRate: number;
  channels: number;
  bitrate: number;
}

/** The probe must come back at least this similar to be trusted (silence or garbage gives ~0). */
const MIN_SIMILARITY = 0.5;

/**
 * Resolves to the delay in samples (at `sampleRate`) that the given codec chain introduces, or null when the
 * browser cannot decode the codec or the probe does not come back intact (then the caller falls back to a known
 * value).
 */
export async function measureEncoderDelay(options: EncoderDelayOptions): Promise<number | null> {
  const { codec, sampleRate, channels, bitrate } = options;
  const key = `${codec}/${sampleRate}/${channels}/${bitrate}`;
  const cached = cache.get(key);
  if (cached !== undefined) return cached;
  const length = Math.round(sampleRate * 0.5);
  const probe = probeSignal(length, sampleRate);
  const planar = new Float32Array(length * channels);
  for (let c = 0; c < channels; c++) planar.set(probe, c * length);
  const isMp4 = codec === 'aac';
  const output = new Output({
    format: isMp4 ? new Mp4OutputFormat({ fastStart: 'in-memory' }) : new WebMOutputFormat(),
    target: new BufferTarget(),
  });
  const source = new AudioSampleSource({ codec, quality: new Quality({ bitrate }) });
  output.addAudioTrack(source);
  await output.start();
  const probeSample = new AudioSample({ data: planar, format: 'f32-planar', numberOfChannels: channels, sampleRate, timestamp: 0 });
  try {
    await source.add(probeSample);
  } finally {
    probeSample.close();
  }
  source.close();
  await output.finalize();
  const bytes = output.target.buffer;
  if (!bytes) return null;
  const input = new Input({ source: new BufferSource(bytes), formats: [isMp4 ? MP4 : WEBM] });
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track || !(await track.canDecode())) return null;
    const decoded = new Float32Array(length + sampleRate);
    const sink = new AudioSampleSink(track);
    let plane: Float32Array | null = null;
    for await (const sample of sink.samples()) {
      try {
        const frames = sample.numberOfFrames;
        const offset = Math.round(sample.timestamp * sample.sampleRate);
        if (sample.sampleRate !== sampleRate) return null;
        if (!plane || plane.length < frames) plane = new Float32Array(frames);
        sample.copyTo(plane, { planeIndex: 0, format: 'f32-planar' });
        const start = Math.max(0, offset);
        const skip = start - offset;
        const count = Math.min(frames - skip, decoded.length - start);
        if (count > 0) decoded.set(plane.subarray(skip, skip + count), start);
      } finally {
        sample.close();
      }
    }
    const { lag, similarity } = findLagDetailed(probe, decoded.subarray(0, length), Math.round(sampleRate * 0.1));
    console.debug(`[encoderDelay] ${key}: lag ${lag} samples, similarity ${similarity.toFixed(3)}`);
    if (similarity < MIN_SIMILARITY) return null;
    cache.set(key, lag);
    return lag;
  } finally {
    input.dispose();
  }
}
