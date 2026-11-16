import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

// base "./": the demo is served under a tunnel prefix (…/proxy/<port>/), so every asset and API path stays relative.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": "/src" } },
  server: {
    // During `npm run dev`, forward api/* to the demo relay.
    proxy: { "/api": "http://127.0.0.1:8002" },
  },
})
