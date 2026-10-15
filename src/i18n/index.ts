/** Language detection, switching, message lookup and locale-aware formatting. */
import { en } from './en.ts';
import { ja, type MessageKey } from './ja.ts';

export type Lang = 'ja' | 'en';
export const LANGS: readonly Lang[] = ['ja', 'en'];

const dictionaries: Record<Lang, Record<MessageKey, string>> = { ja, en };
const STORAGE_KEY = 'bgm-buster.lang';
const listeners = new Set<(lang: Lang) => void>();

let current: Lang = detectLanguage();

function readStored(): Lang | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === 'ja' || v === 'en' ? v : null;
  } catch {
    return null;
  }
}

export function detectLanguage(): Lang {
  const stored = readStored();
  if (stored) return stored;
  const nav = typeof navigator !== 'undefined' ? navigator.language : '';
  return nav.toLowerCase().startsWith('ja') ? 'ja' : 'en';
}

export function getLang(): Lang {
  return current;
}

export function setLang(lang: Lang): void {
  if (lang === current) return;
  current = lang;
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // storage unavailable: the choice lasts for this page view
  }
  applyDocumentLanguage();
  for (const fn of listeners) fn(lang);
}

export function onLangChange(fn: (lang: Lang) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Updates <html lang> and document.title for the current language. */
export function applyDocumentLanguage(): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = current;
  document.title = t('app.title');
}

export type Params = Record<string, string | number>;

export function t(key: MessageKey, params?: Params, lang: Lang = current): string {
  const text = dictionaries[lang][key] ?? dictionaries.ja[key] ?? key;
  if (!params) return text;
  return text.replace(/\{(\w+)\}/g, (m, name: string) => (name in params ? String(params[name]) : m));
}

/** "0:42", "12:03", "1:02:03" (language independent). */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = h > 0 ? String(m).padStart(2, '0') : String(m);
  return `${h > 0 ? `${h}:` : ''}${mm}:${String(sec).padStart(2, '0')}`;
}

/** Rounded duration in words: "20秒" / "20 sec", "3分" / "3 min". */
export function formatDurationWords(seconds: number, lang: Lang = current): string {
  const jaUnits = { second: '秒', minute: '分', hour: '時間' } as const;
  const unit = (value: number, u: 'second' | 'minute' | 'hour'): string => {
    if (lang === 'ja') return `${value}${jaUnits[u]}`; // not every ICU build ships Japanese unit names
    return new Intl.NumberFormat('en-US', { style: 'unit', unit: u, unitDisplay: 'short', maximumFractionDigits: 0 }).format(value);
  };
  if (seconds < 60) return unit(Math.max(5, Math.round(seconds / 5) * 5), 'second');
  if (seconds < 3600) return unit(Math.max(1, Math.round(seconds / 60)), 'minute');
  return unit(Math.round(seconds / 3600), 'hour');
}

/** "31 MB", "1.2 GB" */
export function formatBytes(bytes: number, lang: Lang = current): string {
  const locale = lang === 'ja' ? 'ja-JP' : 'en-US';
  const units = [
    { unit: 'gigabyte', value: 1e9 },
    { unit: 'megabyte', value: 1e6 },
    { unit: 'kilobyte', value: 1e3 },
  ] as const;
  for (const u of units) {
    if (bytes >= u.value) {
      const v = bytes / u.value;
      return new Intl.NumberFormat(locale, { style: 'unit', unit: u.unit, unitDisplay: 'short', maximumFractionDigits: v < 10 ? 1 : 0 }).format(v);
    }
  }
  return new Intl.NumberFormat(locale, { style: 'unit', unit: 'byte', unitDisplay: 'short' }).format(bytes);
}

export function formatPercent(fraction: number, lang: Lang = current): string {
  const locale = lang === 'ja' ? 'ja-JP' : 'en-US';
  return new Intl.NumberFormat(locale, { style: 'percent', maximumFractionDigits: 0 }).format(Math.min(1, Math.max(0, fraction)));
}

export function formatDimensions(width: number, height: number, lang: Lang = current): string {
  const locale = lang === 'ja' ? 'ja-JP' : 'en-US';
  const nf = new Intl.NumberFormat(locale, { useGrouping: false });
  return `${nf.format(width)}×${nf.format(height)}`;
}
