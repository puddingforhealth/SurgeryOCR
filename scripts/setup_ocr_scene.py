#!/usr/bin/env python3
"""Create an isolated scene OCR environment using the existing project PyTorch."""

from pathlib import Path
import shutil
import subprocess
import sys


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Scene OCR experiments require Python 3.11 or newer")
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
        )
    try:
        import torch
        import onnx
    except ImportError as error:
        raise SystemExit(
            "Run the project setup first, then launch this script with that Python interpreter"
        ) from error
    print(f"Inheriting PyTorch {torch.__version__}, ONNX {onnx.__version__}")
    environment = root / ".venv-ocr-scene"
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if not python.exists():
        subprocess.run(
            [
                uv,
                "venv",
                "--system-site-packages",
                "--python",
                sys.executable,
                str(environment),
            ],
            check=True,
        )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(root / "config/ocr_scene_requirements.in"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "import rapidocr, torch, onnx, onnxruntime; print('Scene OCR environment ready; CUDA:', torch.cuda.is_available())",
        ],
        check=True,
    )
    print(f"Run experiments with: {python}")


if __name__ == "__main__":
    main()
