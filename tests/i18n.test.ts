import { describe, expect, it } from 'vitest';
import { en } from '../src/i18n/en.ts';
import { ja } from '../src/i18n/ja.ts';
import { formatBytes, formatClock, formatDurationWords, t } from '../src/i18n/index.ts';

describe('dictionaries', () => {
  it('have exactly the same keys', () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(ja).sort());
  });

  it('use the same placeholders in both languages', () => {
    const placeholders = (s: string): string[] => (s.match(/\{\w+\}/g) ?? []).sort();
    for (const key of Object.keys(ja) as (keyof typeof ja)[]) {
      expect(placeholders(en[key]), key).toEqual(placeholders(ja[key]));
    }
  });

  it('have no empty strings', () => {
    for (const dict of [ja, en]) for (const [k, v] of Object.entries(dict)) expect(v.trim().length, k).toBeGreaterThan(0);
  });
});

describe('t()', () => {
  it('substitutes placeholders', () => {
    expect(t('progress.remaining', { time: '20秒' }, 'ja')).toBe('残り約20秒');
    expect(t('progress.remaining', { time: '20 sec' }, 'en')).toBe('About 20 sec left');
    expect(t('drop.formats', { max: '10分' }, 'ja')).toContain('10分');
  });
});

describe('formatting', () => {
  it('formats clock times', () => {
    expect(formatClock(42)).toBe('0:42');
    expect(formatClock(723)).toBe('12:03');
    expect(formatClock(3723)).toBe('1:02:03');
  });

  it('formats durations in words per language', () => {
    expect(formatDurationWords(600, 'ja')).toMatch(/10\s?分/);
    expect(formatDurationWords(600, 'en')).toMatch(/10 min/);
    expect(formatDurationWords(22, 'en')).toMatch(/20 sec/);
  });

  it('formats sizes', () => {
    expect(formatBytes(30_903_032, 'en')).toMatch(/31 MB/);
    expect(formatBytes(1_234_567_890, 'ja')).toMatch(/1\.2 GB/);
  });
});
