import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** Single-file target.
 *
 * Differs from the main build in one way that matters: the output format is
 * IIFE, not ESM. A `<script type="module">` is refused by Chrome over
 * `file://` (module scripts are fetched with CORS, and a file URL has a null
 * origin), so an ESM single-file bundle would open to a blank page — exactly
 * the situation the single-file build exists to avoid.
 */
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "dist-single",
    assetsDir: "assets",
    sourcemap: false,
    rollupOptions: {
      output: { format: "iife", inlineDynamicImports: true, entryFileNames: "assets/bundle.js" },
    },
  },
});
