import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The api binds to loopback and has no auth, so the dev server proxies rather than
// exposing it to the browser as a second origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
})
