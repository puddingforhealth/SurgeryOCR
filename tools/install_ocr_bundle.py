#!/usr/bin/env python3
"""Install or verify a local OCR model bundle without downloading private weights."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_BUNDLE = ROOT / "models" / "trained_ocr"


def install(source, destination):
    from utils.ocr_runtime import file_sha256, load_bundle

    source, manifest = load_bundle(source)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    installed = json.loads(json.dumps(manifest))
    for key, entry in manifest["artifacts"].items():
        name = key + "-" + entry["sha256"][:16] + Path(entry["file"]).suffix
        target = destination / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            shutil.copy2(source / entry["file"], temporary)
            if file_sha256(temporary) != entry["sha256"]:
                raise ValueError("Copied OCR artifact hash mismatch")
            temporary.replace(target)
        if file_sha256(target) != entry["sha256"]:
            raise ValueError(f"Existing installed artifact is corrupt: {target}")
        installed["artifacts"][key]["file"] = name
    temporary = destination / "manifest.json.part"
    temporary.write_text(json.dumps(installed, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / "manifest.json")
    return load_bundle(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Verified output of build_ocr_bundle.py"
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Verify hashes and runtime prerequisites without changing files",
    )
    args = parser.parse_args(argv)
    if args.doctor:
        import importlib

        setup = (
            r".\scripts\setup.ps1"
            if sys.platform == "win32"
            else "bash scripts/setup.sh"
        )
        if sys.version_info < (3, 11):
            raise SystemExit(
                f"Backend requires Python 3.11+. Interpreter: {sys.executable}. Run {setup}."
            )
        missing = []
        for name in (
            "torch",
            "torchvision",
            "easyocr",
            "onnxruntime",
            "cv2",
            "av",
            "PIL",
            "numpy",
            "tqdm",
        ):
            try:
                importlib.import_module(name)
            except (ImportError, OSError, RuntimeError) as exc:
                missing.append(f"{name} ({exc})")
        if missing:
            raise SystemExit(
                f"Backend dependencies unavailable in {sys.executable}: "
                + ", ".join(missing)
                + f". Run {setup} from the repository root and use its .venv interpreter."
            )
        from utils.ocr_runtime import load_bundle

        try:
            root, manifest = load_bundle(args.destination)
        except FileNotFoundError as error:
            raise SystemExit(
                "OCR bundle missing. Install it with: python tools/install_ocr_bundle.py --source BUNDLE_DIR"
            ) from error
        import torch
        import onnxruntime

        print(f"OCR bundle {manifest['version']}: verified at {root}")
        print(
            f"PyTorch {torch.__version__}; CUDA {torch.cuda.is_available()}; ONNX Runtime {onnxruntime.__version__}"
        )
        return
    if args.source is None:
        parser.error("--source is required unless --doctor is specified")
    root, manifest = install(args.source, args.destination)
    print(f"Installed OCR bundle {manifest['version']}: {root}")


if __name__ == "__main__":
    main()
