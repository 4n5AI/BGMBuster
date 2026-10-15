/**
 * Sample-rate conversion and channel down-mixing with OfflineAudioContext (main thread only).
 * Down-mixing to stereo follows the Web Audio "speakers" interpretation (e.g. 5.1 -> L/R with centre and
 * surrounds folded in).
 */

export function toAudioBuffer(channels: readonly Float32Array[], sampleRate: number): AudioBuffer {
  const buffer = new AudioBuffer({ numberOfChannels: channels.length, length: channels[0].length, sampleRate });
  channels.forEach((ch, i) => buffer.copyToChannel(ch as Float32Array<ArrayBuffer>, i));
  return buffer;
}

export function fromAudioBuffer(buffer: AudioBuffer): Float32Array[] {
  return Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i).slice());
}

/**
 * Converts `channels` at `sampleRate` to `targetRate` with at most `maxChannels` channels.
 * Returns the input unchanged when nothing needs to change.
 */
export async function resampleChannels(
  channels: readonly Float32Array[],
  sampleRate: number,
  targetRate: number,
  maxChannels = 2,
): Promise<Float32Array[]> {
  const outChannels = Math.min(channels.length, maxChannels);
  if (sampleRate === targetRate && channels.length === outChannels) return channels.map((c) => c);
  const length = channels[0].length;
  if (length === 0) return Array.from({ length: outChannels }, () => new Float32Array(0));
  const outLength = Math.ceil((length * targetRate) / sampleRate);
  const context = new OfflineAudioContext({ numberOfChannels: outChannels, length: outLength, sampleRate: targetRate });
  const source = context.createBufferSource();
  source.buffer = toAudioBuffer(channels, sampleRate);
  source.channelCountMode = 'explicit';
  source.channelCount = Math.min(channels.length, 32);
  source.channelInterpretation = 'speakers';
  source.connect(context.destination);
  source.start(0);
  const rendered = await context.startRendering();
  return fromAudioBuffer(rendered);
}
