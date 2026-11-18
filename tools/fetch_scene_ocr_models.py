#!/usr/bin/env python3
"""Download pinned scene OCR weights and verify every artifact before use."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch_artifact(item, destination):
    expected = item["sha256"]
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing file has the wrong SHA-256: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    hasher, total = hashlib.sha256(), 0
    with (
        urllib.request.urlopen(item["url"], timeout=60) as response,
        temporary.open("wb") as target,
    ):
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > 160 * 1024 * 1024:
                raise ValueError("Artifact exceeds the 160 MiB download budget")
            hasher.update(block)
            target.write(block)
    if hasher.hexdigest() != expected:
        raise ValueError(f"Download SHA-256 mismatch: {temporary}")
    temporary.replace(destination)


def main(argv=None):
    registry = json.loads(
        (ROOT / "config/ocr_scene_models.json").read_text(encoding="utf-8")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(registry["models"]), default="v6-small"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, item in registry["models"][args.model].items():
        suffix = {"onnx": ".onnx", "torch": ".pth", "dictionary": "-dictionary.txt"}[
            kind
        ]
        destination = args.output / (args.model + suffix)
        fetch_artifact(item, destination)
        print(f"Verified {kind}: {destination}")


if __name__ == "__main__":
    main()
