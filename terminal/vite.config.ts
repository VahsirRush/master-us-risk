import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` keeps the build portable: it works from a GitHub Pages project
// subpath, from a plain `file://` open, and from `vite preview`, without a
// rebuild per target.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", assetsDir: "assets", sourcemap: false },
});
