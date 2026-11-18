#!/usr/bin/env python3
"""Package selected numeric recognizers with hashes and deployment provenance."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_runtime import file_sha256, load_bundle  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine12-weights", type=Path, required=True)
    parser.add_argument(
        "--machine12-route",
        choices=["conservative", "single_numeric"],
        default="conservative",
    )
    parser.add_argument("--machine12-evidence", type=Path, required=True)
    parser.add_argument("--machine3-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Include a portable CUDA inference model",
    )
    parser.add_argument("--scene-pretrained", type=Path)
    parser.add_argument("--scene-dictionary", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new bundle directory")
    if args.torchscript and not (args.scene_pretrained and args.scene_dictionary):
        parser.error("--torchscript needs --scene-pretrained and --scene-dictionary")
    plan = json.loads((args.machine3_fit / "plan.json").read_text(encoding="utf-8"))
    result = json.loads((args.machine3_fit / "result.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    artifacts = {}
    for key, source, name in [
        ("machine12", args.machine12_weights, "machine12.pth"),
        ("machine3_onnx", args.machine3_fit / "numeric-scene.onnx", "machine3.onnx"),
    ]:
        shutil.copy2(source, args.output / name)
        artifacts[key] = {"file": name, "sha256": file_sha256(args.output / name)}
    if args.torchscript:
        import torch
        from utils.scene_ocr import load_scene_model

        torch.set_num_threads(2)
        model = load_scene_model(
            plan["arguments"]["architecture"],
            args.scene_pretrained,
            args.scene_dictionary,
        )
        model.load_state_dict(
            torch.load(
                args.machine3_fit / "best.pth", map_location="cpu", weights_only=True
            )
        )
        model.eval()
        traced = torch.jit.trace(model, torch.zeros(1, 3, 48, 320), check_trace=True)
        with torch.inference_mode():
            for size in [1, 3, 8]:
                probe = torch.randn(size, 3, 48, 320)
                torch.testing.assert_close(
                    traced(probe), model(probe), rtol=1e-4, atol=1e-5
                )
        traced.save(str(args.output / "machine3.pt"))
        artifacts["machine3_torchscript"] = {
            "file": "machine3.pt",
            "sha256": file_sha256(args.output / "machine3.pt"),
        }
    manifest = {
        "schema_version": 1,
        "version": args.version,
        "scene_characters": "+-0123456789. ",
        "artifacts": artifacts,
        "machine12_evidence_sha256": file_sha256(args.machine12_evidence),
        "machine3_fit_plan_sha256": file_sha256(args.machine3_fit / "plan.json"),
        "machine3_dataset_sha256": plan["dataset_sha256"],
        "machine3_training_samples": len(plan["train"]),
        "machine3_validation_samples": len(plan["val"]),
        "machine3_independent_test_samples": len(plan["test"]),
        "machine3_selected_epoch": result["best_epoch"],
        "annotation_status": "assistant-reviewed; not independently human verified",
        "acceptance_scope": "Research integration; development source videos are not untouched acceptance data",
        "license_notice": "Retain PaddleOCR/RapidOCR and EasyOCR upstream attribution. Private images and fonts are not included.",
    }
    if args.machine12_route == "single_numeric":
        manifest.update(schema_version=2, machine12_route=args.machine12_route)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    load_bundle(args.output)
    print("Verified bundle:", args.output)


if __name__ == "__main__":
    main()
