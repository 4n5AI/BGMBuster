/** Development page: media input/output round trip without separation (see dev/media-test.html). */
import { LIMITS, MODEL } from '../config.ts';
import { resampleChannels } from '../audio/resample.ts';
import { encodeWav16 } from '../audio/wav.ts';
import { exportVideo, ExportError } from '../export/mux.ts';
import { decodeAudioTrack } from '../media/decode.ts';
import { MediaError, openMedia, type MediaInfo } from '../media/probe.ts';
import { ALL_FORMATS, BufferSource, Input } from 'mediabunny';
import { findLag } from '../export/encoderDelay.ts';

const logEl = document.getElementById('log') as HTMLPreElement;
const lines: string[] = [];
function log(line: string): void {
  lines.push(line);
  logEl.textContent = lines.join('\n');
  console.log(line);
}

export interface RoundTripResult {
  info?: MediaInfo;
  decoded?: { channels: number; sampleRate: number; frames: number; firstTimestamp: number; seconds: number };
  resampled?: { channels: number; frames: number; seconds: number };
  output?: { bytes: number; mimeType: string; extension: string; audioCodec: string; audioBitrate: number; seconds: number; encoderDelay: number | null };
  wavBytes?: number;
  /** Offset of the exported audio against the decoded source audio, measured in this browser (ms; + = late). */
  syncMs?: { start: number; end: number } | null;
  error?: { kind: string; message: string };
}

/** Envelope of the high-passed signal (first difference, rectified, 2 ms average) for click-based alignment. */
function clickEnvelope(x: Float32Array, sr: number): Float32Array {
  const k = Math.max(1, Math.round(sr * 0.002));
  const env = new Float32Array(x.length);
  let acc = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - (i > 0 ? x[i - 1] : 0));
    acc += d;
    if (i >= k) acc -= Math.abs(x[i - k] - (i - k > 0 ? x[i - k - 1] : 0));
    env[i] = acc / k;
  }
  return env;
}

function offsetMs(a: Float32Array, b: Float32Array, sr: number): number {
  const ea = clickEnvelope(a, sr);
  const eb = clickEnvelope(b, sr);
  return (findLag(ea, eb, Math.round(sr * 0.5)) / sr) * 1000;
}

declare global {
  interface Window {
    __roundTrip: (file: File) => Promise<RoundTripResult>;
    __roundTripResult?: RoundTripResult;
    __outputBytes?: Uint8Array;
    __wavBytes?: Uint8Array;
  }
}

async function roundTrip(file: File): Promise<RoundTripResult> {
  const result: RoundTripResult = {};
  try {
    log(`--- ${file.name} (${(file.size / 1e6).toFixed(2)} MB)`);
    const t0 = performance.now();
    const media = await openMedia(file, LIMITS.maxDurationSeconds);
    result.info = media.info;
    log(`opened: ${JSON.stringify(media.info)}`);
    const decoded = await decodeAudioTrack(media.audioTrack);
    const seconds = (performance.now() - t0) / 1000;
    result.decoded = { channels: decoded.channels.length, sampleRate: decoded.sampleRate, frames: decoded.channels[0].length, firstTimestamp: decoded.firstTimestamp, seconds };
    log(`decoded: ${JSON.stringify(result.decoded)}`);
    const t1 = performance.now();
    const model = await resampleChannels(decoded.channels, decoded.sampleRate, MODEL.sampleRate, 2);
    result.resampled = { channels: model.length, frames: model[0].length, seconds: (performance.now() - t1) / 1000 };
    log(`resampled to ${MODEL.sampleRate}: ${JSON.stringify(result.resampled)}`);
    // back to the original rate (what the exporter gets after separation)
    const back = await resampleChannels(model, MODEL.sampleRate, decoded.sampleRate, 2);
    const wav = encodeWav16(back, decoded.sampleRate);
    result.wavBytes = wav.byteLength;
    window.__wavBytes = new Uint8Array(wav);
    if (media.videoTrack) {
      const t2 = performance.now();
      const out = await exportVideo({
        input: media.input,
        videoTrack: media.videoTrack,
        container: media.info.container,
        audio: back,
        sampleRate: decoded.sampleRate,
        audioOffsetSeconds: decoded.firstTimestamp,
        originalBitrate: media.info.audio.bitrate,
        onProgress: (f) => {
          if (Math.round(f * 100) % 25 === 0) log(`export ${(f * 100).toFixed(0)}%`);
        },
      });
      result.output = {
        bytes: out.bytes?.byteLength ?? 0,
        mimeType: out.mimeType,
        extension: out.extension,
        audioCodec: out.audioCodec,
        audioBitrate: out.audioBitrate,
        seconds: (performance.now() - t2) / 1000,
        encoderDelay: out.encoderDelay,
      };
      if (out.bytes) window.__outputBytes = new Uint8Array(out.bytes);
      log(`exported: ${JSON.stringify(result.output)}`);
      // decode the output again and compare with the source audio at both ends
      if (out.bytes) {
        try {
          const check = new Input({ source: new BufferSource(out.bytes), formats: ALL_FORMATS });
          const track = await check.getPrimaryAudioTrack();
          if (track && (await track.canDecode())) {
            const again = await decodeAudioTrack(track);
            const src = decoded.channels[0];
            const dst = again.channels[0];
            const lead = Math.max(0, Math.round(again.firstTimestamp * again.sampleRate));
            const srcStart = Math.max(0, Math.round(decoded.firstTimestamp * decoded.sampleRate));
            const w = Math.min(src.length - srcStart, dst.length - lead, decoded.sampleRate * 2);
            const n = Math.min(src.length - srcStart, dst.length - lead);
            const s0 = src.subarray(srcStart, srcStart + w);
            const d0 = dst.subarray(lead, lead + w);
            const s1 = src.subarray(srcStart + n - w, srcStart + n);
            const d1 = dst.subarray(lead + n - w, lead + n);
            result.syncMs = { start: offsetMs(s0, d0, decoded.sampleRate), end: offsetMs(s1, d1, decoded.sampleRate) };
            log(`sync (decoded in this browser): start ${result.syncMs.start.toFixed(2)} ms, end ${result.syncMs.end.toFixed(2)} ms`);
          } else {
            result.syncMs = null;
            log('sync: cannot decode the output in this browser');
          }
          check.dispose();
        } catch (e) {
          log(`sync check failed: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    } else {
      log('audio only: WAV output');
    }
    media.input.dispose();
  } catch (e) {
    const kind = e instanceof MediaError || e instanceof ExportError ? e.kind : e instanceof Error ? e.name : 'unknown';
    result.error = { kind, message: e instanceof Error ? e.message : String(e) };
    log(`ERROR ${kind}: ${result.error.message}`);
  }
  window.__roundTripResult = result;
  return result;
}

window.__roundTrip = roundTrip;
document.getElementById('file')?.addEventListener('change', (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (f) void roundTrip(f);
});

const src = new URLSearchParams(location.search).get('src');
if (src) {
  void (async () => {
    const res = await fetch(src);
    const blob = await res.blob();
    await roundTrip(new File([blob], src.split('/').pop() ?? 'input', { type: blob.type }));
  })();
}
