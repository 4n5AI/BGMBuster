/**
 * Preview playback: a muted media element shows the picture (and is the transport the user controls);
 * the sound comes from Web Audio so that "original / music removed / music only" and the music level can
 * change instantly without stopping. Play, pause and seeking follow the element's events; drift is corrected.
 *
 * Graph: mix ──gMix──┐
 *        voice ─gVoice┼─► master ─► destination        (voice = speech + sfx)
 *        music ─gMusic┘
 *   original      : gMix 1, gVoice 0, gMusic 0
 *   removed (A)   : gMix 0, gVoice 1, gMusic g
 *   removed (B)   : gMix 1, gVoice 0, gMusic -(1 - g)
 *   music only    : gMix 0, gVoice 0, gMusic 1
 */
import type { RemixMethod } from './remix.ts';

export type ListenMode = 'original' | 'removed' | 'music';

export interface PlayerBuffers {
  mix: AudioBuffer;
  voice: AudioBuffer;
  music: AudioBuffer;
}

const DRIFT_TOLERANCE = 0.06; // seconds

export class SyncedPlayer {
  private context: AudioContext | null = null;
  private gains: { mix: GainNode; voice: GainNode; music: GainNode; master: GainNode } | null = null;
  private sources: AudioBufferSourceNode[] = [];
  private startedAt = 0; // context time when sources started
  private startOffset = 0; // media time at which they started
  private readonly media: HTMLMediaElement;
  private readonly buffers: PlayerBuffers;
  private mode: ListenMode = 'removed';
  private method: RemixMethod;
  private level = 0;
  private driftTimer: number | null = null;
  private readonly onPlay = (): void => void this.start();
  private readonly onPause = (): void => this.stop();
  private readonly onSeeked = (): void => {
    if (!this.media.paused) void this.start();
  };
  private readonly onRateChange = (): void => {
    for (const s of this.sources) s.playbackRate.value = this.media.playbackRate;
  };

  constructor(media: HTMLMediaElement, buffers: PlayerBuffers, method: RemixMethod, level: number) {
    this.media = media;
    this.buffers = buffers;
    this.method = method;
    this.level = level;
    media.muted = true;
    media.addEventListener('play', this.onPlay);
    media.addEventListener('pause', this.onPause);
    media.addEventListener('seeked', this.onSeeked);
    media.addEventListener('ended', this.onPause);
    media.addEventListener('ratechange', this.onRateChange);
  }

  private ensureContext(): AudioContext {
    if (this.context) return this.context;
    const context = new AudioContext({ sampleRate: this.buffers.mix.sampleRate });
    const master = context.createGain();
    master.connect(context.destination);
    const mix = context.createGain();
    const voice = context.createGain();
    const music = context.createGain();
    mix.connect(master);
    voice.connect(master);
    music.connect(master);
    this.context = context;
    this.gains = { mix, voice, music, master };
    this.applyGains(0);
    return context;
  }

  private applyGains(ramp = 0.02): void {
    if (!this.context || !this.gains) return;
    const g = Math.min(1, Math.max(0, this.level));
    let values: [number, number, number];
    if (this.mode === 'original') values = [1, 0, 0];
    else if (this.mode === 'music') values = [0, 0, 1];
    else values = this.method === 'A' ? [0, 1, g] : [1, 0, -(1 - g)];
    const t = this.context.currentTime;
    const targets = [this.gains.mix, this.gains.voice, this.gains.music];
    targets.forEach((node, i) => {
      node.gain.cancelScheduledValues(t);
      if (ramp > 0) {
        node.gain.setValueAtTime(node.gain.value, t);
        node.gain.linearRampToValueAtTime(values[i], t + ramp);
      } else {
        node.gain.setValueAtTime(values[i], t);
      }
    });
  }

  setMode(mode: ListenMode): void {
    this.mode = mode;
    this.applyGains();
  }

  setMusicLevel(level: number): void {
    this.level = level;
    this.applyGains();
  }

  setMethod(method: RemixMethod): void {
    this.method = method;
    this.applyGains();
  }

  get currentTime(): number {
    return this.media.currentTime;
  }

  private async start(): Promise<void> {
    const context = this.ensureContext();
    if (context.state === 'suspended') await context.resume();
    this.stopSources();
    const offset = Math.max(0, this.media.currentTime);
    const now = context.currentTime + 0.02;
    const make = (buffer: AudioBuffer, gain: GainNode): AudioBufferSourceNode => {
      const s = context.createBufferSource();
      s.buffer = buffer;
      s.playbackRate.value = this.media.playbackRate;
      s.connect(gain);
      if (offset < buffer.duration) s.start(now, offset);
      return s;
    };
    if (!this.gains) return;
    this.sources = [make(this.buffers.mix, this.gains.mix), make(this.buffers.voice, this.gains.voice), make(this.buffers.music, this.gains.music)];
    this.startedAt = now;
    this.startOffset = offset;
    if (this.driftTimer === null) this.driftTimer = window.setInterval(() => this.checkDrift(), 500);
  }

  private checkDrift(): void {
    if (!this.context || this.media.paused || this.sources.length === 0) return;
    const audioTime = this.startOffset + (this.context.currentTime - this.startedAt) * this.media.playbackRate;
    if (Math.abs(audioTime - this.media.currentTime) > DRIFT_TOLERANCE) void this.start();
  }

  private stopSources(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        // not started
      }
      s.disconnect();
    }
    this.sources = [];
  }

  private stop(): void {
    this.stopSources();
    if (this.driftTimer !== null) {
      window.clearInterval(this.driftTimer);
      this.driftTimer = null;
    }
  }

  dispose(): void {
    this.stop();
    this.media.removeEventListener('play', this.onPlay);
    this.media.removeEventListener('pause', this.onPause);
    this.media.removeEventListener('seeked', this.onSeeked);
    this.media.removeEventListener('ended', this.onPause);
    this.media.removeEventListener('ratechange', this.onRateChange);
    void this.context?.close();
    this.context = null;
    this.gains = null;
  }
}
