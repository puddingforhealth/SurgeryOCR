#!/usr/bin/env bash
# One-shot setup for a Linux machine (cloud GPU job, VM, container).
# Idempotent: safe to re-run. After it finishes:
#   gcloud auth login --no-browser          # once per machine
#   python tools/cloud_videos.py doctor
#   python tools/cloud_videos.py pull --all --run
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
SDK_DIR="$PWD/tools/google-cloud-sdk"

echo "== system libraries (OpenCV needs libGL)"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  if ! ldconfig -p | grep -q libGL.so.1; then
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq libgl1 libglib2.0-0 curl
  fi
fi

echo "== python packages"
$PY -m pip install --upgrade pip -q
$PY -m pip install -r requirements.txt -q

echo "== google cloud sdk"
if ! command -v gcloud >/dev/null 2>&1; then
  if [ ! -x "$SDK_DIR/bin/gcloud" ]; then
    mkdir -p tools
    ARCH=$(uname -m); case "$ARCH" in aarch64|arm64) ARCH=arm ;; *) ARCH=x86_64 ;; esac
    URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-${ARCH}.tar.gz"
    curl -fsSL "$URL" | tar -xz -C tools
    "$SDK_DIR/install.sh" --quiet --usage-reporting false --path-update false >/dev/null
  fi
  export PATH="$SDK_DIR/bin:$PATH"
  echo "   installed to $SDK_DIR  (add to PATH: export PATH=\"$SDK_DIR/bin:\$PATH\")"
fi
gcloud --version | head -1

echo "== check"
$PY tools/cloud_videos.py doctor || true

cat <<EOF

Next:
  export PATH="$SDK_DIR/bin:\$PATH"      # only if gcloud was installed above
  gcloud auth login --no-browser           # paste the printed command on a machine with a browser
  python tools/cloud_videos.py pull --all --run --upload-results
EOF
