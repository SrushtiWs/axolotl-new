#!/usr/bin/env bash
# Run the backend and the frontend together.
#
#   ./dev.sh
#
# Backend  -> http://localhost:8000   (FastAPI, POST /generate)
# Frontend -> http://localhost:5173   (Vite dev server)
#
# Ctrl-C stops both.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="$ROOT/backend/.venv"

if [ ! -x "$VENV/bin/uvicorn" ]; then
  echo "Creating backend virtualenv..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/backend/requirements.txt"
fi

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd "$ROOT/frontend" && npm install)
fi

if [ ! -f "$ROOT/frontend/.env.local" ]; then
  echo "VITE_API_BASE_URL=http://localhost:8000" > "$ROOT/frontend/.env.local"
fi

cleanup() {
  trap - INT TERM EXIT
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

"$VENV/bin/uvicorn" app:app --app-dir "$ROOT/backend" --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

wait -n "$BACKEND_PID" "$FRONTEND_PID"
