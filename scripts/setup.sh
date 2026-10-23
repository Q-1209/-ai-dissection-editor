#!/usr/bin/env bash
# One-command setup for macOS / Linux.
#   bash scripts/setup.sh            (add --skip-model to skip the model download)
# Creates .venv (CPU PyTorch + requirements), downloads Qwen2.5-0.5B-Instruct into models/, and runs npm install.
# The app picks up .venv automatically, so afterwards just run: npm run electron:dev
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SKIP_MODEL=0
[ "${1:-}" = "--skip-model" ] && SKIP_MODEL=1

step() { printf '\n==> %s\n' "$1"; }

step "Checking Python (3.10-3.13) and Node.js"
# The newest Python may not have wheels for every dependency yet, so prefer a stable version
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.10-3.13 is required: https://www.python.org/downloads/"; exit 1; }
"$PY" --version
node --version

VPY=.venv/bin/python
if [ -x "$VPY" ]; then
  "$VPY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)'     || { echo ".venv was made with an unsupported Python. Delete .venv and run this script again."; exit 1; }
else
  step "Creating .venv"
  "$PY" -m venv .venv
fi

step "Installing PyTorch and requirements"
"$VPY" -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then
  "$VPY" -m pip install torch            # macOS wheels include Apple Silicon support
else
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$VPY" -m pip install -r requirements.txt huggingface_hub

if [ "$SKIP_MODEL" = 0 ]; then
  DIR="models/${MODEL##*/}"
  if [ ! -f "$DIR/config.json" ]; then
    step "Downloading $MODEL (about 1 GB)"
    "$VPY" -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$DIR')"
  fi
fi

step "Installing app dependencies (npm install)"
npm install

step "Done. Start the app with:  npm run electron:dev"
echo "Then click 'Open model folder' and choose the folder under models/"
