#!/usr/bin/env python3
"""Create an isolated CPU OCR benchmark environment with pinned dependencies."""

from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
        )
    env = root / ".venv-ocr-benchmark"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        subprocess.run([uv, "venv", "--python", sys.executable, str(env)], check=True)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(root / "config/ocr_benchmark_requirements.in"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "import cv2, numpy, onnxruntime; from PIL import Image; print('CPU OCR benchmark environment ready')",
        ],
        check=True,
    )
    print(f"Run benchmarks with: {python}")


if __name__ == "__main__":
    main()
