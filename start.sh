#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
}
trap cleanup INT TERM

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "Error: python3 / python not found in PATH." >&2
  exit 1
fi

# --prod: build the frontend once and let the backend serve it (one process, http://localhost:8000).
if [ "${1:-}" = "--prod" ]; then
  echo "Building frontend..."
  (cd "$SCRIPT_DIR/frontend" && npm run build)
  echo "Starting backend (serving the built app)..."
  cd "$SCRIPT_DIR/backend"
  echo ""
  echo "App: http://localhost:8000"
  echo ""
  exec "$PYTHON" serve.py
fi

echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
"$PYTHON" serve.py &
BACKEND_PID=$!

echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

wait
