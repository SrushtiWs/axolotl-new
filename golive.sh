#!/usr/bin/env bash
#
# Put the app on a public URL with Cloudflare Tunnel.
#
#   ./golive.sh
#
# Starts, in order: the backend, a tunnel to it, the frontend pointed at that
# tunnel, and a tunnel to the frontend. Prints both public URLs and stays in the
# foreground — Ctrl-C stops everything.
#
# The one step this exists to remove is copying the backend URL by hand into
# frontend/.env.local. A quick tunnel mints a new hostname on every run, so that
# value cannot be written down in advance; typing it manually has put a literal
# placeholder in the file three times, and a placeholder host is NXDOMAIN, which
# the browser reports as "Failed to fetch". Here the URL is captured from
# cloudflared's own output and written for you.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGS="$ROOT/.golive"
mkdir -p "$LOGS"

PIDS=()

cleanup() {
  echo
  echo "Stopping… (the public URLs die with these processes)"
  for pid in "${PIDS[@]:-}"; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null
  done
  exit 0
}

trap cleanup INT TERM

command -v cloudflared >/dev/null || {
  echo "cloudflared is not installed.  brew install cloudflared"
  exit 1
}

# A quick tunnel's hostname only lives as long as its process, so any tunnel
# still running from a previous session is pointing at nothing useful.
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 2

# ---- 1. backend ------------------------------------------------------------
if lsof -ti:8000 >/dev/null 2>&1; then
  echo "[1/4] backend already on :8000"
else
  echo "[1/4] starting backend on :8000"
  "$ROOT/backend/.venv/bin/uvicorn" app:app --app-dir "$ROOT/backend" \
    --host 127.0.0.1 --port 8000 > "$LOGS/backend.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 40); do
    curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null || {
  echo "  backend did not come up — see $LOGS/backend.log"
  exit 1
}

# ---- 2. tunnel to the backend ---------------------------------------------
echo "[2/4] opening backend tunnel"

: > "$LOGS/cf-backend.log"
cloudflared tunnel --url http://127.0.0.1:8000 > "$LOGS/cf-backend.log" 2>&1 &
PIDS+=($!)

BACKEND_URL=""
for _ in $(seq 1 40); do
  BACKEND_URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOGS/cf-backend.log" 2>/dev/null | head -1)
  [ -n "$BACKEND_URL" ] && break
  sleep 2
done

[ -n "$BACKEND_URL" ] || { echo "  no backend tunnel URL — see $LOGS/cf-backend.log"; cleanup; }

echo "      $BACKEND_URL"
printf '      /health through it: '
curl -s -m 25 -o /dev/null -w '%{http_code}\n' "$BACKEND_URL/health"

# ---- 3. frontend, pointed at that tunnel ----------------------------------
# Written before Vite starts: Vite reads .env.local once, at boot, so a value
# written afterwards is ignored until it restarts.
echo "[3/4] pointing the frontend at it and starting Vite"

echo "VITE_API_BASE_URL=$BACKEND_URL" > "$ROOT/frontend/.env.local"

lsof -ti:5173 | xargs kill -9 2>/dev/null
sleep 1

(cd "$ROOT/frontend" && npm run dev > "$LOGS/frontend.log" 2>&1) &
PIDS+=($!)

for _ in $(seq 1 40); do
  curl -sf -m 2 http://127.0.0.1:5173/ >/dev/null 2>&1 && break
  sleep 1
done

# ---- 4. tunnel to the frontend --------------------------------------------
echo "[4/4] opening frontend tunnel"

: > "$LOGS/cf-frontend.log"
cloudflared tunnel --url http://127.0.0.1:5173 > "$LOGS/cf-frontend.log" 2>&1 &
PIDS+=($!)

FRONTEND_URL=""
for _ in $(seq 1 40); do
  FRONTEND_URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOGS/cf-frontend.log" 2>/dev/null | head -1)
  [ -n "$FRONTEND_URL" ] && break
  sleep 2
done

[ -n "$FRONTEND_URL" ] || { echo "  no frontend tunnel URL — see $LOGS/cf-frontend.log"; cleanup; }

echo
echo "─────────────────────────────────────────────────────────────"
echo "  OPEN THIS:  $FRONTEND_URL"
echo "  backend  :  $BACKEND_URL"
echo "─────────────────────────────────────────────────────────────"
printf '  CORS check: '
curl -s -o /dev/null -m 25 -w '%{http_code}\n' -X OPTIONS "$BACKEND_URL/segment" \
  -H "Origin: $FRONTEND_URL" -H "Access-Control-Request-Method: POST"
echo
echo "  Clean Room takes 90-300s. Cloudflare cuts any request at 100s,"
echo "  so it may fail over this URL — it always works on localhost:5173."
echo
echo "  Ctrl-C stops everything. Both URLs die when you do."

wait
