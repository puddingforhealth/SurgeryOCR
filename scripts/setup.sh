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

# PyPI's Linux torch targets the newest CUDA, which an older driver rejects ("driver too
# old"). pytorch.org hosts one index per CUDA version, so pick the newest one the driver
# runs, and replace an installed torch that was built for a newer CUDA than that.
version_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }
DRIVER_CUDA=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA[A-Z ]*Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
fi
TORCH_CUDA=$("$VPY" -c "import torch; print(torch.version.cuda or 'cpu')" 2>/dev/null || true)
if [ -n "$DRIVER_CUDA" ]; then
  echo "   driver supports CUDA $DRIVER_CUDA; installed torch: ${TORCH_CUDA:-none}"
  if [ -n "$TORCH_CUDA" ] && { [ "$TORCH_CUDA" = "cpu" ] || ! version_le "$TORCH_CUDA" "$DRIVER_CUDA"; }; then
    "$VPY" -m pip uninstall -y -q torch torchvision
    TORCH_CUDA=""
  fi
fi
if [ -z "$TORCH_CUDA" ]; then
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    "${PIP[@]}" --index-url "$TORCH_INDEX_URL" torch torchvision
  elif [ -n "$DRIVER_CUDA" ]; then
    # Only torch itself comes from pytorch.org: that index sends its nvidia-* CUDA
    # dependencies to pypi.nvidia.com, which some cloud firewalls block. The same
    # nvidia-* versions are on PyPI, so the second install resolves them there.
    TORCH_TAG=""
    for tag in cu130 cu129 cu128 cu126 cu124 cu121 cu118; do
      ver="${tag:2:-1}.${tag: -1}"
      version_le "$ver" "$DRIVER_CUDA" || continue
      if "${PIP[@]}" --no-deps --index-url "https://download.pytorch.org/whl/$tag" torch torchvision; then
        TORCH_TAG=$tag; break
      fi
    done
    [ -z "$TORCH_TAG" ] || "${PIP[@]}" torch torchvision
    [ -n "$TORCH_TAG" ] || { echo "error: no torch build found for CUDA $DRIVER_CUDA; set TORCH_INDEX_URL" >&2; exit 1; }
    echo "   torch from the $TORCH_TAG index"
  else
    "${PIP[@]}" torch torchvision
  fi
fi
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
