/** Decoding the first audio track to planar float samples. */
import { AudioSampleSink, type InputAudioTrack } from 'mediabunny';

export interface DecodedAudio {
  /** One Float32Array per channel, at `sampleRate`. */
  channels: Float32Array[];
  sampleRate: number;
  /** Presentation time (seconds) of the first sample; usually 0 or slightly off. */
  firstTimestamp: number;
}

/**
 * Decodes the whole track. `onProgress` receives the decoded seconds so far.
 * Samples are laid out so that index 0 corresponds to `firstTimestamp`; gaps between samples are filled with
 * silence so that sample index / sampleRate + firstTimestamp is the presentation time.
 */
export async function decodeAudioTrack(track: InputAudioTrack, onProgress?: (seconds: number, total: number) => void): Promise<DecodedAudio> {
  const sampleRate = await track.getSampleRate();
  const numberOfChannels = await track.getNumberOfChannels();
  const duration = await track.computeDuration();
  const firstTimestamp = await track.getFirstTimestamp();
  const capacity = Math.ceil((duration - firstTimestamp) * sampleRate) + sampleRate; // headroom for rounding
  const channels = Array.from({ length: numberOfChannels }, () => new Float32Array(capacity));
  const sink = new AudioSampleSink(track);
  let maxEnd = 0;
  let plane: Float32Array | null = null;
  for await (const sample of sink.samples()) {
    try {
      const frames = sample.numberOfFrames;
      const offset = Math.round((sample.timestamp - firstTimestamp) * sampleRate);
      const start = Math.max(0, offset);
      if (start + frames > capacity) break;
      if (!plane || plane.length < frames) plane = new Float32Array(frames);
      const chCount = Math.min(numberOfChannels, sample.numberOfChannels);
      for (let c = 0; c < chCount; c++) {
        sample.copyTo(plane, { planeIndex: c, format: 'f32-planar' });
        channels[c].set(plane.subarray(0, frames), start);
      }
      maxEnd = Math.max(maxEnd, start + frames);
      onProgress?.(Math.min(duration, sample.timestamp + sample.duration), duration);
    } finally {
      sample.close();
    }
  }
  const length = maxEnd;
  // Opus in MP4: the file's edit list expresses the codec pre-skip, so Mediabunny reports the first sample at
  // -preSkip, but the WebCodecs Opus decoder has already dropped those samples (measured: keeping the negative
  // offset makes the audio early by exactly the pre-skip). For Opus the decoded stream therefore starts at 0.
  // AAC keeps the negative offset: its priming samples really are in the decoded stream and must be skipped.
  const codec = await track.getCodec();
  const reportedFirst = codec === 'opus' && firstTimestamp < 0 ? 0 : firstTimestamp;
  return {
    channels: channels.map((ch) => ch.subarray(0, length)),
    sampleRate,
    firstTimestamp: reportedFirst,
  };
}
