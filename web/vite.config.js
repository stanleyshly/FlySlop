import { defineConfig } from 'vite';
export default defineConfig({
  root: 'src',
  base: './',
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { outDir: '..', emptyOutDir: false, assetsDir: 'assets' },
});
