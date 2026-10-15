/**
 * Audio encoders add priming samples (AAC-LC: 1024) and not every encoder/muxer pair signals them, which
 * would leave the audio late against the picture. This measures the end-to-end delay of "encode -> mux ->
 * demux -> decode" for the codec in use once per session, so the exporter can shift its timestamps.
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

function probeSignal(length: number): Float32Array {
  // deterministic noise burst: broadband, so the cross-correlation peak is sharp
  const x = new Float32Array(length);
  let state = 987654321;
  const start = Math.floor(length * 0.2);
  const end = Math.floor(length * 0.8);
  for (let i = start; i < end; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    x[i] = (state / 4294967296 - 0.5) * 0.8;
  }
  return x;
}

/** Lag of `b` relative to `a` in samples (positive: b is late), searched within ±maxLag. */
export function findLag(a: Float32Array, b: Float32Array, maxLag: number): number {
  let bestLag = 0;
  let best = -Infinity;
  const n = Math.min(a.length, b.length);
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
  return bestLag;
}

/**
 * Resolves to the delay in samples (at `sampleRate`) that the given codec chain introduces, or null when the
 * browser cannot decode the codec (then the caller falls back to a known value).
 */
export async function measureEncoderDelay(codec: AudioCodec, sampleRate: number, channels: number, bitrate: number): Promise<number | null> {
  const key = `${codec}/${sampleRate}/${channels}/${bitrate}`;
  const cached = cache.get(key);
  if (cached !== undefined) return cached;
  const length = Math.round(sampleRate * 0.5);
  const probe = probeSignal(length);
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
  await source.add(new AudioSample({ data: planar, format: 'f32-planar', numberOfChannels: channels, sampleRate, timestamp: 0 }));
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
    const lag = findLag(probe, decoded.subarray(0, length), Math.round(sampleRate * 0.1));
    cache.set(key, lag);
    return lag;
  } finally {
    input.dispose();
  }
}
