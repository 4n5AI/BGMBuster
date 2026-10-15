import { defineConfig } from 'vite';

// GitHub Pages serves the site under /<repository>/; the deploy workflow sets BASE_PATH accordingly.
const base = process.env.BASE_PATH ?? '/';

export default defineConfig({
  base,
  build: {
    target: 'es2022',
    sourcemap: false,
    rollupOptions: {
      input: {
        main: 'index.html',
        bench: 'bench.html',
        // development-only page for the media round-trip check
        ...(process.env.BGM_DEV_PAGES ? { mediaTest: 'dev/media-test.html' } : {}),
      },
    },
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      // Match GitHub Pages: no cross-origin isolation, so WASM multithreading stays off in development too.
    },
  },
});
