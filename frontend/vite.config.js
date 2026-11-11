import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// /api is forwarded to FastAPI, so model URLs like /api/inventory/... work without CORS.
// The build contains the app and the headless renderer page (render.html); FastAPI serves the
// build under /renderer/, so relative asset paths are used.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' }
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        render: resolve(import.meta.dirname, 'render.html')
      }
    }
  }
});
