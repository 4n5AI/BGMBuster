/** Handing files to the user: the File System Access API when available, otherwise a download link. */

export function replaceExtension(fileName: string, suffix: string, extension: string): string {
  const dot = fileName.lastIndexOf('.');
  const stem = dot > 0 ? fileName.slice(0, dot) : fileName;
  return `${stem}${suffix}${extension}`;
}

export interface SaveTarget {
  /** Present when the browser let the user pick a file; stream the output into it. */
  writable?: FileSystemWritableFileStream;
  /** Call with the bytes when no writable is available. */
  finish(bytes: ArrayBuffer | Blob, mimeType: string): Promise<void>;
  cancel(): Promise<void>;
}

type SavePicker = (options: { suggestedName: string; types: { description: string; accept: Record<string, string[]> }[] }) => Promise<FileSystemFileHandle>;

/**
 * Asks where to save (Chrome/Edge) before a long export starts, so the result can be streamed to disk.
 * Resolves to null when the user cancelled the picker.
 */
export async function chooseSaveTarget(suggestedName: string, mimeType: string, extension: string): Promise<SaveTarget | null> {
  const picker = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  if (picker) {
    try {
      const handle = await picker.call(window, { suggestedName, types: [{ description: suggestedName, accept: { [mimeType]: [extension] } }] });
      const writable = await handle.createWritable();
      return {
        writable,
        finish: async () => undefined,
        cancel: async () => {
          await writable.abort().catch(() => undefined);
        },
      };
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return null;
      // Fall back to a download when the picker is unavailable (e.g. not a user gesture).
    }
  }
  return {
    finish: async (bytes, type) => {
      const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type });
      downloadBlob(blob, suggestedName);
    },
    cancel: async () => undefined,
  };
}

export function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.rel = 'noopener';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
