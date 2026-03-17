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
      // Use the page's hostname so HMR works over LAN (not just localhost).
      // Without this, Vite tries to connect the websocket to localhost even
      // when the page is loaded via a LAN IP, causing HMR failures and
      // full-page reloads -- especially painful on iOS Safari where tab
      // freezing kills the websocket and Vite's reconnect triggers a reload.
      host: undefined, // auto-detect from the request
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
