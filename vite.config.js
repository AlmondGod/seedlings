import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: {
    proxy: {
      "/agent-action": "http://127.0.0.1:8787"
    }
  }
});
