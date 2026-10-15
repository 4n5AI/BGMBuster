/** Downloads the model once, keeps it in Cache Storage, and verifies its SHA-256 every time. */

export class ModelError extends Error {
  readonly kind: 'download' | 'corrupt';

  constructor(kind: 'download' | 'corrupt', message: string) {
    super(message);
    this.name = 'ModelError';
    this.kind = kind;
  }
}

export interface ModelExpectation {
  sha256: string;
  bytes: number;
}

export type DownloadProgress = (loaded: number, total: number) => void;

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function openCache(name: string): Promise<Cache | null> {
  try {
    if (typeof caches === 'undefined') return null;
    return await caches.open(name);
  } catch {
    return null;
  }
}

async function readWithProgress(response: Response, expectedBytes: number, onProgress?: DownloadProgress): Promise<ArrayBuffer> {
  const headerLength = Number(response.headers.get('Content-Length') ?? '0');
  const total = headerLength > 0 ? headerLength : expectedBytes;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, total);
    return buffer;
  }
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, Math.max(total, loaded));
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  return out.buffer;
}

/**
 * Resolves to the verified model bytes. The first call downloads and caches; later calls come from the cache.
 * Throws ModelError('download') when the network fails and ModelError('corrupt') when the hash never matches.
 */
export async function loadModelBytes(url: string, cacheName: string, expected: ModelExpectation, onProgress?: DownloadProgress): Promise<ArrayBuffer> {
  const cache = await openCache(cacheName);
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        const buffer = await hit.arrayBuffer();
        if ((await sha256Hex(buffer)) === expected.sha256) {
          onProgress?.(buffer.byteLength, buffer.byteLength);
          return buffer;
        }
        await cache.delete(url);
      }
    } catch {
      // A broken cache entry is not fatal; fall through to the network.
    }
  }
  let response: Response;
  try {
    response = await fetch(url, { cache: 'no-cache' });
  } catch (e) {
    throw new ModelError('download', `fetch failed: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (!response.ok) throw new ModelError('download', `HTTP ${response.status} for ${url}`);
  let buffer: ArrayBuffer;
  try {
    buffer = await readWithProgress(response, expected.bytes, onProgress);
  } catch (e) {
    throw new ModelError('download', `read failed: ${e instanceof Error ? e.message : String(e)}`);
  }
  const hash = await sha256Hex(buffer);
  if (hash !== expected.sha256) {
    throw new ModelError('corrupt', `SHA-256 mismatch: got ${hash.slice(0, 16)}…, expected ${expected.sha256.slice(0, 16)}… (${buffer.byteLength} bytes)`);
  }
  if (cache) {
    try {
      await cache.put(
        url,
        new Response(buffer.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buffer.byteLength) },
        }),
      );
    } catch {
      // Storage may be full or disallowed; the download still succeeded.
    }
  }
  return buffer;
}
