/** Rendering the one-column, three-step screen from the store state. */
import { SyncedPlayer, type ListenMode } from '../audio/player.ts';
import { LIMITS, MODEL, REPO_URL } from '../config.ts';
import { formatBytes, formatClock, formatDimensions, formatDurationWords, formatPercent, getLang, LANGS, setLang, t, type Lang } from '../i18n/index.ts';
import type { MessageKey } from '../i18n/ja.ts';
import { append, clear, h } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import type { Pipeline, StemName } from './pipeline.ts';
import type { AppState, ErrorAction, Store } from './state.ts';

const ACCEPT = '.mp4,.m4v,.mov,.webm,.wav,.mp3,.m4a,video/mp4,video/quicktime,video/webm,audio/wav,audio/x-wav,audio/mpeg,audio/mp4,audio/x-m4a';

export class View {
  private readonly root: HTMLElement;
  private readonly store: Store;
  private readonly pipeline: Pipeline;
  private media: HTMLMediaElement | null = null;
  private player: SyncedPlayer | null = null;
  private playerFor: AppState['output'] = null;
  private fileInput: HTMLInputElement;
  private dialog: HTMLDialogElement | null = null;

  constructor(root: HTMLElement, store: Store, pipeline: Pipeline) {
    this.root = root;
    this.store = store;
    this.pipeline = pipeline;
    this.fileInput = h('input', { type: 'file', accept: ACCEPT, class: 'visually-hidden', tabindex: -1, 'aria-hidden': 'true' });
    this.fileInput.addEventListener('change', () => {
      const f = this.fileInput.files?.[0];
      if (f) void this.pipeline.selectFile(f);
      this.fileInput.value = '';
    });
    store.subscribe((state, prev) => this.render(state, prev));
    this.render(store.get(), store.get());
  }

  private render(state: AppState, prev: AppState): void {
    if (state.phase === 'idle' && prev.phase !== 'idle') this.disposePlayer();
    if (state.output !== this.playerFor) this.disposePlayer();
    clear(this.root);
    append(this.root, [this.fileInput, this.header(), this.step1(state), this.step2(state), this.step3(state), this.footer()]);
    if (state.phase === 'done' && state.output && !this.player) this.createPlayer(state);
    this.player?.setMode(state.listen);
    this.player?.setMusicLevel(state.musicLevel);
    this.player?.setSfxLevel(state.sfxLevel);
    this.player?.setMethod(state.method);
  }

  // ---------- header / footer ----------

  private header(): HTMLElement {
    const lang = getLang();
    const buttons = LANGS.map((l: Lang) =>
      h('button', { type: 'button', 'aria-pressed': String(l === lang), lang: l, onclick: () => setLang(l) }, t(l === 'ja' ? 'lang.ja' : 'lang.en')),
    );
    return h(
      'header',
      { class: 'header' },
      h('div', null, h('h1', null, t('app.name')), h('p', { class: 'tagline' }, t('app.tagline'))),
      h('div', { class: 'lang-switch', role: 'group', 'aria-label': t('lang.label') }, ...buttons),
    );
  }

  private footer(): HTMLElement {
    return h(
      'footer',
      { class: 'footer' },
      h('span', null, t('footer.license')),
      h('button', { type: 'button', onclick: () => this.openCredits() }, t('footer.credits')),
      h('a', { href: REPO_URL, target: '_blank', rel: 'noopener' }, t('footer.github')),
    );
  }

  private openCredits(): void {
    this.dialog?.remove();
    const dialog = h(
      'dialog',
      { 'aria-labelledby': 'credits-title' },
      h('h2', { id: 'credits-title' }, t('credits.title')),
      h('p', null, t('credits.model')),
      h('p', null, h('a', { href: 'https://arxiv.org/abs/2110.09958', target: '_blank', rel: 'noopener' }, t('credits.paper'))),
      h('p', null, t('credits.libraries')),
      h('p', null, h('a', { href: `${REPO_URL}/blob/main/THIRD_PARTY_NOTICES.md`, target: '_blank', rel: 'noopener' }, t('credits.notices'))),
      h('p', null, h('button', { type: 'button', class: 'btn', onclick: () => dialog.close() }, t('credits.close'))),
    );
    document.body.append(dialog);
    this.dialog = dialog;
    dialog.addEventListener('close', () => dialog.remove());
    dialog.showModal();
  }

  // ---------- step 1 ----------

  private step1(state: AppState): HTMLElement {
    const active = state.phase === 'idle' || state.phase === 'loaded' || state.phase === 'error';
    const drop = h(
      'div',
      {
        class: 'drop',
        ondragover: (ev: Event) => {
          ev.preventDefault();
          drop.setAttribute('data-over', '');
        },
        ondragleave: () => drop.removeAttribute('data-over'),
        ondrop: (ev: Event) => {
          ev.preventDefault();
          drop.removeAttribute('data-over');
          const f = (ev as DragEvent).dataTransfer?.files?.[0];
          if (f) void this.pipeline.selectFile(f);
        },
      },
      icon('upload', 28),
      h('p', null, t('drop.hint')),
      h('p', { class: 'muted' }, t('drop.or')),
      h('button', { type: 'button', class: 'btn btn-primary', onclick: () => this.fileInput.click() }, t('drop.button')),
      h('p', { class: 'muted' }, t('drop.formats', { max: formatDurationWords(LIMITS.maxDurationSeconds) })),
    );
    return h(
      'section',
      { class: 'step', 'data-inactive': !active, 'aria-labelledby': 'step1-title' },
      h('h2', { id: 'step1-title' }, h('span', { class: 'step-number', 'aria-hidden': 'true' }, '1'), t('step1.title')),
      drop,
      h('p', { class: 'privacy' }, icon('lock', 16), h('span', null, t('app.privacy'))),
    );
  }

  // ---------- step 2 ----------

  private step2(state: AppState): HTMLElement {
    const active = state.phase === 'loaded' || state.phase === 'preparing' || state.phase === 'processing' || (state.phase === 'error' && !!state.file);
    const children: (HTMLElement | null)[] = [];
    if (state.info) {
      const info = state.info;
      const detail = info.video ? formatDimensions(info.video.width, info.video.height) : t('file.audioOnly');
      const sep = t('file.summary', { name: '', duration: '', detail: '' }).trim().charAt(0) || '·';
      children.push(
        h(
          'p',
          { class: 'file-summary' },
          icon(info.video ? 'film' : 'music', 18),
          h('span', { class: 'file-name' }, info.fileName),
          h('span', { class: 'file-detail' }, ` ${sep} ${formatClock(info.duration)} ${sep} ${detail}`),
        ),
      );
      for (const note of state.notes) children.push(h('p', { class: 'note' }, t(note)));
    }
    const busy = state.phase === 'preparing' || state.phase === 'processing';
    if (state.phase === 'loaded') {
      children.push(
        h(
          'div',
          { class: 'btn-row' },
          h('button', { type: 'button', class: 'btn btn-primary btn-lg', onclick: () => void this.pipeline.run() }, t('action.remove')),
          h('button', { type: 'button', class: 'btn', onclick: () => this.fileInput.click() }, t('action.chooseFile')),
        ),
      );
      if (state.canceled) children.push(h('p', { class: 'note', 'aria-live': 'polite' }, t('progress.canceled')));
    }
    if (busy && state.progress) children.push(this.progress(state));
    if (busy) {
      children.push(h('div', { class: 'btn-row' }, h('button', { type: 'button', class: 'btn', onclick: () => this.pipeline.cancel() }, t('action.cancel'))));
    }
    if (state.phase === 'error' && state.error && (state.phase !== 'error' || !state.output)) children.push(this.errorBox(state));
    return h(
      'section',
      { class: 'step', 'data-inactive': !active, 'aria-labelledby': 'step2-title' },
      h('h2', { id: 'step2-title' }, h('span', { class: 'step-number', 'aria-hidden': 'true' }, '2'), t('step2.title')),
      ...children,
    );
  }

  private progress(state: AppState): HTMLElement {
    const p = state.progress!;
    const labels: Record<typeof p.stage, string> = {
      model: t('progress.model', { size: formatBytes(MODEL.bytes) }),
      read: t('progress.read'),
      separate: t('progress.separate'),
      write: t('progress.write'),
    };
    const pct = p.fraction === null ? null : formatPercent(p.fraction);
    const right: string[] = [];
    if (p.stage === 'model' && p.loaded !== undefined && p.total) right.push(`${formatBytes(p.loaded)} / ${formatBytes(p.total)}`);
    if (p.stage === 'separate' && p.etaSeconds !== undefined && p.fraction && p.fraction > 0.02) right.push(t('progress.remaining', { time: formatDurationWords(p.etaSeconds) }));
    const bar = h('div', { class: 'progress-bar', 'data-indeterminate': p.fraction === null, role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': p.fraction === null ? null : Math.round(p.fraction * 100), 'aria-label': labels[p.stage] }, h('div', { style: p.fraction === null ? '' : `width:${(p.fraction * 100).toFixed(1)}%` }));
    return h(
      'div',
      { class: 'progress' },
      bar,
      h('div', { class: 'progress-text', 'aria-live': 'polite' }, h('span', null, `${labels[p.stage]}${pct ? ` ${pct}` : ''}`), h('span', null, right.join(' '))),
      state.tech.wasmSlow && p.stage === 'separate' ? h('p', { class: 'note' }, t('progress.slow')) : null,
    );
  }

  private errorBox(state: AppState): HTMLElement {
    const error = state.error!;
    const actionButton = (a: ErrorAction): HTMLElement => {
      const labels: Record<ErrorAction, MessageKey> = { retry: 'action.retry', reload: 'action.reload', chooseFile: 'action.chooseFile', saveAudio: 'action.saveAudio' };
      const handlers: Record<ErrorAction, () => void> = {
        retry: () => void (state.output ? this.store.set({ error: null }) : this.pipeline.run()),
        reload: () => location.reload(),
        chooseFile: () => this.fileInput.click(),
        saveAudio: () => void this.pipeline.saveAudio(),
      };
      return h('button', { type: 'button', class: a === error.actions[0] ? 'btn btn-primary' : 'btn', onclick: handlers[a] }, t(labels[a]));
    };
    return h(
      'div',
      { class: 'error', role: 'alert' },
      icon('warning', 22),
      h('div', null, h('p', null, t(error.key, error.params)), error.actions.length ? h('div', { class: 'btn-row' }, ...error.actions.map(actionButton)) : null),
    );
  }

  // ---------- step 3 ----------

  private step3(state: AppState): HTMLElement {
    const active = state.phase === 'done';
    const children: (HTMLElement | null)[] = [];
    if (active && state.output && state.fileUrl) {
      children.push(this.preview(state));
      children.push(this.listenControls(state));
      children.push(this.saveButtons(state));
      if (state.error) children.push(this.errorBox(state));
      children.push(this.details(state));
      children.push(h('p', { class: 'note' }, t('note.quality')));
      children.push(h('p', { class: 'note' }, t('note.rights')));
      children.push(h('div', { class: 'btn-row' }, h('button', { type: 'button', class: 'btn', onclick: () => this.pipeline.reset() }, t('action.another'))));
    }
    return h(
      'section',
      { class: 'step', 'data-inactive': !active, 'aria-labelledby': 'step3-title' },
      h('h2', { id: 'step3-title' }, h('span', { class: 'step-number', 'aria-hidden': 'true' }, '3'), t('step3.title')),
      ...children,
    );
  }

  private preview(state: AppState): HTMLElement {
    const hasVideo = !!state.info?.video;
    if (!this.media) {
      this.media = hasVideo
        ? h('video', { controls: true, muted: true, playsinline: true, preload: 'metadata', src: state.fileUrl!, controlslist: 'nodownload' })
        : h('audio', { controls: true, muted: true, preload: 'metadata', src: state.fileUrl!, controlslist: 'nodownload' });
      this.media.muted = true;
    }
    return h('div', { class: 'preview', 'data-audio': !hasVideo }, this.media);
  }

  private createPlayer(state: AppState): void {
    const out = state.output!;
    if (!this.media) return;
    const buffers = { mix: out.mix, speech: out.speech, sfx: out.sfx, music: out.music };
    this.player = new SyncedPlayer(this.media, buffers, state.method, state.musicLevel, state.sfxLevel);
    this.player.setMode(state.listen);
    this.playerFor = out;
  }

  private disposePlayer(): void {
    this.player?.dispose();
    this.player = null;
    this.playerFor = null;
    if (this.media) {
      this.media.pause();
      this.media.removeAttribute('src');
      this.media.load();
      this.media = null;
    }
  }

  private listenControls(state: AppState): HTMLElement {
    const modes: { mode: ListenMode; key: MessageKey }[] = [
      { mode: 'original', key: 'listen.original' },
      { mode: 'removed', key: 'listen.removed' },
      { mode: 'music', key: 'listen.music' },
    ];
    const sliderDisabled = state.listen !== 'removed';
    const makeSlider = (id: string, value: number, onInput: (v: number) => void): HTMLInputElement =>
      h('input', {
        type: 'range',
        id,
        min: 0,
        max: 100,
        step: 1,
        value: Math.round(value * 100),
        disabled: sliderDisabled,
        'aria-valuetext': formatPercent(value),
        oninput: (ev: Event) => onInput(Number((ev.target as HTMLInputElement).value) / 100),
      });
    const slider = makeSlider('music-level', state.musicLevel, (v) => this.store.set({ musicLevel: v }));
    const sfxSlider = makeSlider('sfx-level', state.sfxLevel, (v) => this.store.set({ sfxLevel: v }));
    return h(
      'div',
      { class: 'listen' },
      h(
        'div',
        { role: 'group', 'aria-label': t('listen.label') },
        h('span', { class: 'muted', style: 'margin-right:10px' }, t('listen.label')),
        h(
          'div',
          { class: 'segmented' },
          ...modes.map((m) => h('button', { type: 'button', 'aria-pressed': String(state.listen === m.mode), onclick: () => this.store.set({ listen: m.mode }) }, t(m.key))),
        ),
      ),
      h('div', { class: 'slider', 'data-disabled': sliderDisabled }, h('label', { for: 'music-level' }, t('volume.label')), slider, h('span', { 'aria-hidden': 'true' }, formatPercent(state.musicLevel))),
      h('div', { class: 'slider', 'data-disabled': sliderDisabled }, h('label', { for: 'sfx-level' }, t('effects.label')), sfxSlider, h('span', { 'aria-hidden': 'true' }, formatPercent(state.sfxLevel))),
      h('p', { class: 'note' }, t('note.effects')),
    );
  }

  private saveButtons(state: AppState): HTMLElement {
    const saving = state.saving !== null;
    const label = (key: MessageKey, kind: AppState['saving']): string =>
      state.saving === kind ? `${t('save.preparing')}${state.saveProgress !== null ? ` ${formatPercent(state.saveProgress)}` : ''}` : t(key);
    const buttons: HTMLElement[] = [];
    if (this.pipeline.hasVideo()) {
      buttons.push(h('button', { type: 'button', class: 'btn btn-primary', disabled: saving, onclick: () => void this.pipeline.saveVideo() }, icon('download', 18), label('save.video', 'video')));
    }
    buttons.push(h('button', { type: 'button', class: this.pipeline.hasVideo() ? 'btn' : 'btn btn-primary', disabled: saving, onclick: () => void this.pipeline.saveAudio() }, icon('download', 18), label('save.audio', 'audio')));
    return h('div', { class: 'btn-row', 'aria-live': 'polite' }, ...buttons);
  }

  private details(state: AppState): HTMLElement {
    const stems: { stem: StemName; key: MessageKey }[] = [
      { stem: 'speech', key: 'stem.dialogue' },
      { stem: 'sfx', key: 'stem.effects' },
      { stem: 'music', key: 'stem.music' },
    ];
    const tech = state.tech;
    const rows: [string, string][] = [];
    if (tech.backend) {
      const parts = [t(tech.backend === 'webgpu' ? 'tech.backend.webgpu' : 'tech.backend.wasm')];
      if (tech.adapter) parts.push(tech.adapter);
      if (tech.selfTest) parts.push(`self-test ${tech.selfTest}`);
      if (tech.backendReason) parts.push(tech.backendReason);
      rows.push([t('tech.backend'), parts.join(' · ')]);
    }
    rows.push([t('tech.model'), `${MODEL.id} (${formatBytes(MODEL.bytes)})`]);
    if (state.info) rows.push([t('tech.input'), `${state.info.formatName} · ${state.info.audio.codec ?? '?'} ${state.info.audio.sampleRate} Hz ${state.info.audio.channels} ch${state.info.video ? ` · ${state.info.video.codec ?? '?'}` : ''}`]);
    const times: string[] = [];
    if (tech.modelLoadSeconds !== undefined) times.push(`model ${tech.modelLoadSeconds.toFixed(1)} s`);
    if (tech.decodeSeconds !== undefined) times.push(`decode ${tech.decodeSeconds.toFixed(1)} s`);
    if (tech.separateSeconds !== undefined) times.push(`separate ${tech.separateSeconds.toFixed(1)} s (${state.output?.chunks ?? 0} chunks)`);
    if (tech.exportSeconds !== undefined) times.push(`export ${tech.exportSeconds.toFixed(1)} s`);
    if (times.length) rows.push([t('tech.time'), times.join(' · ')]);
    if (tech.exportCodec) {
      const delay = tech.encoderDelay === undefined ? '' : tech.encoderDelay === null ? ` · ${t('tech.delay', { samples: '1024' })} (${t('tech.delay.default')})` : ` · ${t('tech.delay', { samples: String(tech.encoderDelay) })} (${t('tech.delay.measured')})`;
      rows.push([t('tech.export'), `${tech.exportCodec} ${Math.round((tech.exportBitrate ?? 0) / 1000)} kbps${delay}`]);
    }
    if (state.error?.detail) rows.push([t('tech.error'), state.error.detail]);
    return h(
      'details',
      null,
      h('summary', null, t('details.title')),
      h(
        'div',
        { class: 'stems' },
        ...stems.map((s) => h('button', { type: 'button', class: 'btn', disabled: state.saving !== null, onclick: () => void this.pipeline.saveStem(s.stem) }, icon('download', 18), t(s.key))),
      ),
      h('p', { class: 'muted', style: 'margin-top:12px' }, t('details.tech')),
      h('dl', { class: 'tech' }, ...rows.flatMap(([k, v]) => [h('dt', null, k), h('dd', null, v)])),
    );
  }
}
