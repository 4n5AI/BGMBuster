/** Opening a media file with Mediabunny and reading what we need to know about it. */
import {
  ALL_FORMATS,
  BlobSource,
  Input,
  type InputAudioTrack,
  type InputFormat,
  type InputVideoTrack,
  IsobmffInputFormat,
  MatroskaInputFormat,
  Mp3InputFormat,
  QuickTimeInputFormat,
  WaveInputFormat,
} from 'mediabunny';

export type MediaErrorKind = 'unsupported-format' | 'no-audio' | 'too-long' | 'undecodable-audio';

export class MediaError extends Error {
  readonly kind: MediaErrorKind;

  constructor(kind: MediaErrorKind, message: string) {
    super(message);
    this.name = 'MediaError';
    this.kind = kind;
  }
}

export type ContainerKind = 'mp4' | 'mov' | 'webm' | 'mkv' | 'wav' | 'mp3' | 'other';

export interface VideoInfo {
  codec: string | null;
  width: number;
  height: number;
  rotation: number;
  frameRate: number | null;
}

export interface MediaInfo {
  fileName: string;
  fileSize: number;
  container: ContainerKind;
  formatName: string;
  /** Seconds; the longest track. */
  duration: number;
  video: VideoInfo | null;
  audioTrackCount: number;
  audio: {
    codec: string | null;
    sampleRate: number;
    channels: number;
    bitrate: number | null;
    firstTimestamp: number;
    duration: number;
  };
}

export interface OpenedMedia {
  input: Input;
  info: MediaInfo;
  videoTrack: InputVideoTrack | null;
  audioTrack: InputAudioTrack;
}

export function containerKind(format: InputFormat): ContainerKind {
  if (format instanceof QuickTimeInputFormat) return 'mov';
  if (format instanceof IsobmffInputFormat) return 'mp4';
  if (format instanceof MatroskaInputFormat) return format.name.toLowerCase().includes('webm') ? 'webm' : 'mkv';
  if (format instanceof WaveInputFormat) return 'wav';
  if (format instanceof Mp3InputFormat) return 'mp3';
  return 'other';
}

/** Opens the file and reads its metadata. Throws MediaError for files we cannot use. */
export async function openMedia(file: File, maxDurationSeconds: number): Promise<OpenedMedia> {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  let format: InputFormat;
  try {
    format = await input.getFormat();
  } catch (e) {
    input.dispose();
    throw new MediaError('unsupported-format', e instanceof Error ? e.message : String(e));
  }
  const container = containerKind(format);
  if (container === 'other') {
    input.dispose();
    throw new MediaError('unsupported-format', `container ${format.name} is not supported`);
  }
  const audioTracks = await input.getAudioTracks();
  if (audioTracks.length === 0) {
    input.dispose();
    throw new MediaError('no-audio', 'the file has no audio track');
  }
  const audioTrack = audioTracks[0];
  const duration = await input.computeDuration();
  if (duration > maxDurationSeconds) {
    input.dispose();
    throw new MediaError('too-long', `${duration.toFixed(1)} s exceeds ${maxDurationSeconds} s`);
  }
  if (!(await audioTrack.canDecode())) {
    input.dispose();
    throw new MediaError('undecodable-audio', `cannot decode audio codec ${await audioTrack.getCodec()}`);
  }
  const videoTrack = await input.getPrimaryVideoTrack();
  let video: VideoInfo | null = null;
  if (videoTrack) {
    let frameRate: number | null = null;
    try {
      frameRate = (await videoTrack.computeFrameRateMetrics()).bestGuessFrameRate;
    } catch {
      frameRate = null;
    }
    video = {
      codec: await videoTrack.getCodec(),
      width: await videoTrack.getDisplayWidth(),
      height: await videoTrack.getDisplayHeight(),
      rotation: await videoTrack.getRotation(),
      frameRate,
    };
  }
  const info: MediaInfo = {
    fileName: file.name,
    fileSize: file.size,
    container,
    formatName: format.name,
    duration,
    video,
    audioTrackCount: audioTracks.length,
    audio: {
      codec: await audioTrack.getCodec(),
      sampleRate: await audioTrack.getSampleRate(),
      channels: await audioTrack.getNumberOfChannels(),
      bitrate: (await audioTrack.getAverageBitrate()) ?? (await audioTrack.getBitrate()),
      firstTimestamp: await audioTrack.getFirstTimestamp(),
      duration: await audioTrack.computeDuration(),
    },
  };
  return { input, info, videoTrack, audioTrack };
}
