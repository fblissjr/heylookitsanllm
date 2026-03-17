import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/v1": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
    hmr: {
      // Disable the error overlay to prevent it from triggering reload loops
      // when the HMR websocket reconnects after iOS Safari tab restore.
      overlay: false,
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
