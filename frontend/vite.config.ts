import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // /api/... -> backend FastAPI sin el prefijo (/api/music -> /music)
    // en docker el backend es 'backend:8000'; en local, localhost
    proxy: {
      "/api": {
        target: process.env.BACKEND_URL || "http://localhost:8000",
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
