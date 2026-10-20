#!/usr/bin/env bash
# One-shot setup for a Linux machine (cloud GPU job, VM, container).
# Idempotent: safe to re-run, e.g. after every `git pull`. Installs every Python
# dependency into .venv, so a new shell only needs:
#   source .venv/bin/activate
#
# Environment overrides:
#   PYTHON=python3.11        interpreter the venv is created from (default python3)
#   VENV_DIR=/big/disk/venv  keep the venv on another disk; .venv becomes a symlink to it
#   TORCH_INDEX_URL=<url>    pip index for torch (default PyPI, whose Linux wheels bundle CUDA)
#   SKIP_GCLOUD=1            skip the Google Cloud SDK (not needed when videos arrive another way)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV_LINK="$PWD/.venv"
VENV=${VENV_DIR:-$VENV_LINK}
SDK_DIR="$PWD/tools/google-cloud-sdk"

echo "== system libraries (OpenCV needs libGL, venv needs ensurepip)"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  PKGS=()
  ldconfig -p | grep -q libGL.so.1 || PKGS+=(libgl1 libglib2.0-0)
  command -v curl >/dev/null 2>&1 || PKGS+=(curl)
  "$PY" -c "import ensurepip" 2>/dev/null \
    || PKGS+=("python$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv")
  if [ ${#PKGS[@]} -gt 0 ]; then
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${PKGS[@]}"
  fi
fi

echo "== virtual environment"
if [ "$VENV" != "$VENV_LINK" ]; then
  mkdir -p "$VENV"
  VENV="$(cd "$VENV" && pwd)"
  if [ -e "$VENV_LINK" ] && [ ! -L "$VENV_LINK" ]; then
    echo "error: .venv already exists as a real directory; remove it or unset VENV_DIR" >&2
    exit 1
  fi
  ln -sfn "$VENV" "$VENV_LINK"
fi
# A venv whose interpreter no longer runs (e.g. the base Python was upgraded) is rebuilt.
if ! "$VENV/bin/python" -c "" 2>/dev/null; then
  "$PY" -m venv --clear "$VENV"
fi
VPY="$VENV_LINK/bin/python"
"$VPY" --version
echo "   $VENV"

echo "== python packages"
# No pip cache: container root disks are small, and re-runs skip satisfied requirements anyway.
PIP=("$VPY" -m pip install -q --no-cache-dir)
"${PIP[@]}" --upgrade pip
"${PIP[@]}" ${TORCH_INDEX_URL:+--index-url "$TORCH_INDEX_URL"} torch torchvision
"${PIP[@]}" -r requirements.txt

if [ "${SKIP_GCLOUD:-}" != "1" ]; then
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
fi

echo "== gpu"
# A GPU the driver sees but torch cannot use means the wrong torch build: fail loudly
# instead of letting OCR silently run on CPU.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  "$VPY" -c "import torch, sys; ok = torch.cuda.is_available(); print('   torch', torch.__version__, '->', torch.cuda.get_device_name(0) if ok else 'CUDA NOT AVAILABLE'); sys.exit(0 if ok else 1)" \
    || { echo "error: nvidia-smi sees a GPU but torch cannot; set TORCH_INDEX_URL to a matching CUDA build (see https://pytorch.org)" >&2; exit 1; }
else
  echo "   no NVIDIA GPU visible; OCR will run on CPU"
fi

echo "== check"
"$VPY" tools/cloud_videos.py doctor || true

cat <<EOF

Done. In every new shell:
  source .venv/bin/activate
  python run_pipeline.py --video data/<video>.MP4 --mode ram
EOF
if [ "${SKIP_GCLOUD:-}" != "1" ]; then
  cat <<EOF
To move videos through GCS:
  export PATH="$SDK_DIR/bin:\$PATH"      # only if gcloud was installed above
  gcloud auth login --no-browser           # paste the printed command on a machine with a browser
  python tools/cloud_videos.py pull --all --run --upload-results
EOF
fi
