import './styles.css';
import { Pipeline, browserSupported } from './app/pipeline.ts';
import { Store } from './app/state.ts';
import { View } from './app/view.ts';
import { applyDocumentLanguage, onLangChange, t } from './i18n/index.ts';
import { h } from './ui/dom.ts';
import { icon } from './ui/icons.ts';

const root = document.getElementById('app');
if (!root) throw new Error('#app missing');

applyDocumentLanguage();

if (!browserSupported()) {
  const render = (): void => {
    root.replaceChildren(
      h('header', { class: 'header' }, h('div', null, h('h1', null, t('app.name')), h('p', { class: 'tagline' }, t('app.tagline')))),
      h('div', { class: 'step' }, h('div', { class: 'error', role: 'alert' }, icon('warning', 22), h('p', null, t('error.browser')))),
    );
  };
  render();
  onLangChange(render);
} else {
  const store = new Store();
  const pipeline = new Pipeline(store);
  const view = new View(root, store, pipeline);
  onLangChange(() => {
    applyDocumentLanguage();
    store.set({});
  });
  window.addEventListener('beforeunload', (ev) => {
    const phase = store.get().phase;
    if (phase === 'processing' || phase === 'preparing' || store.get().saving) ev.preventDefault();
  });
  void view;
}
