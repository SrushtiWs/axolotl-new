import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Cloudflare Tunnel mints a fresh hostname on every `cloudflared` run, so
    // the host cannot be written down here. A leading dot matches any
    // subdomain, which is what lets a new quick-tunnel URL work without
    // editing this file. FRONTEND_HOST is for a named tunnel on your own
    // domain; it is simply absent for quick tunnels.
    allowedHosts: [".trycloudflare.com", process.env.FRONTEND_HOST].filter(Boolean),
  },
});
	
