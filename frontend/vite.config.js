import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/predict": "https://stress-detector-api-yhx4.onrender.com",
      "/batch_predict": "https://stress-detector-api-yhx4.onrender.com",
      "/health": "https://stress-detector-api-yhx4.onrender.com",
      "/metadata": "https://stress-detector-api-yhx4.onrender.com",
      "/features": "https://stress-detector-api-yhx4.onrender.com",
      "/api": "https://stress-detector-api-yhx4.onrender.com"
    }
  }
});
