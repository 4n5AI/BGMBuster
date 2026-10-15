/** Fixed facts about the deployment, the model, and the processing limits. */

export const APP_VERSION = '0.1.0';
export const REPO_URL = 'https://github.com/4n5AI/BGMBuster';

/** The separation model shipped with the site (same origin, under /models/). */
export const MODEL = {
  id: 'mrx-v1',
  /** Path relative to the site base. */
  file: 'models/mrx-v1.int8.onnx',
  bytes: 30_903_032,
  sha256: '0791cc73b8174cc9f906b0905161b93ac5b09c5fdda3841619247d087385c4ba',
  sampleRate: 44100,
  sources: ['music', 'speech', 'sfx'] as const,
  license: 'MIT (Copyright (c) 2023 Mitsubishi Electric Research Laboratories)',
  cacheName: 'bgm-buster-models-v1',
} as const;

/** Chunked processing (the upstream model sees whole files; chunks keep memory bounded). */
export const CHUNK = {
  /** Length of one chunk in seconds, at MODEL.sampleRate (20 s deviates least from whole-file processing). */
  seconds: 20,
  /** Overlap between neighbouring chunks in seconds; cross-faded. */
  overlapSeconds: 2,
  crossfade: 'equal-gain' as 'equal-gain' | 'equal-power',
} as const;

/** Upstream separate.py normalises the input to this integrated loudness before inference. */
export const INPUT_TARGET_LUFS = -27;

export const LIMITS = {
  /** Rough upper bound shown in the UI; refined with measurements. */
  maxDurationSeconds: 10 * 60,
} as const;

/**
 * Backend self-test reference: mean mask values of the probe signal (src/separation/selftest.ts) computed with
 * the WebAssembly backend (scripts/bench/selftest-ref.mjs). A backend that deviates by more than `tolerance`
 * (relative) is not used.
 */
export const SELFTEST = {
  reference: {
    m1024: [0.289729249645187, 0.02862918033156291, 0.11483864020241395] as [number, number, number],
    m8192: [0.16376951768570464, 0.037117100353904156, 0.17458748416302752] as [number, number, number],
  },
  tolerance: 0.05,
} as const;

/** Audio encoding when writing video. */
export const EXPORT = {
  aacMinBitrate: 128_000,
  opusBitrate: 128_000,
} as const;
