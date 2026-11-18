# Toolchain

## Optional OCR training and CPU comparison

`tools/ocr_finetune.py` uses the main environment's PyTorch, EasyOCR, NumPy and
Pillow. The verified combination is EasyOCR 1.7.2 / PyTorch 2.8.0; it needs a
local English gen2 weight file. `validate` checks data provenance and hashes;
`train` and `evaluate` write new result folders. `tools/prepare_ocr_job.py` copies
only explicit reviewed inputs into a portable rcs job folder and installs
nothing in a shared remote environment. See [the workflow](ocr-finetuning.md).

`python scripts/setup_ocr_benchmark.py` creates `.venv-ocr-benchmark` using `uv`
and installs exact CPU dependency versions from
`config/ocr_benchmark_requirements.in`. It is idempotent and does not change the
main environment. `tools/benchmark_ocr_onnx.py` uses this environment for local
inference with separately downloaded and hash-verified CTC ONNX weights. No GPU,
hosted inference, or PaddlePaddle installation is required. Remove only the
`.venv-ocr-benchmark` directory to uninstall the optional environment.

For dedicated machine-3 RGB recognizer experiments, run
`python scripts/setup_ocr_scene.py` with the project's Python. It creates
`.venv-ocr-scene` with system-site access to the existing PyTorch/ONNX and pins
additional packages through `config/ocr_scene_requirements.in`. Use
`python tools/fetch_scene_ocr_models.py --model v6-small --output build/scene-models`
for the pinned SHA-256-verified model and dictionary. Installation checks the
runtime imports and CUDA availability. The environment can be removed to
uninstall it; the main environment is unchanged. See
[the full workflow](ocr-machine3-experiments.md) for training, export,
cross-platform commands, all flags and validation boundaries.

Enhanced production OCR uses the private bundle installed by
`python tools/install_ocr_bundle.py --source BUNDLE_DIR`; `--doctor` verifies
hashes and prerequisites. CPU scene inference uses `onnxruntime==1.25.1` on
Python 3.11+ (included in normal and release requirements); CUDA inference uses
the optional packaged TorchScript artifact and existing PyTorch. RapidOCR is
needed only in the isolated training/packaging environment. See
[integrated profiles](ocr-production-integration.md) for model packaging,
configuration, rollback, full-pipeline comparisons and saved viewer layouts.

Everything the pipeline needs, where it lives, and which step uses it.
`python tools/cloud_videos.py doctor` checks all of it and reports what is missing.

| Tool | Purpose | Where | Installed by | Used by |
|---|---|---|---|---|
| Python 3.10+ | runtime | Linux: `.venv/` (gitignored; a symlink when `VENV_DIR` is set). Windows: current Python | Linux: `scripts/setup.sh` creates the venv from `python3` (or `$PYTHON`); Windows: you | everything |
| `opencv-python`, `numpy`, `Pillow` | frame decode, binarize, template matching | pip | `scripts/setup.*` via `requirements.txt` | `run_pipeline.py`, `stage_pattern_analysis.py` |
| OpenCV version pin | `opencv-python` and `opencv-python-headless` (pulled in by EasyOCR) are both pinned to `4.11.0.86`. Both wheels ship the same `cv2/` package, and OpenCV 5.0 decodes the same video to different pixels (every frame, ~3 grey levels on average), which changes some OCR readings. Measured on 2026-10-20: with 4.11, a Linux H100 run matches the Windows run reading for reading. Upgrade only together with new golden baselines | `requirements.txt` | `scripts/setup.sh` reinstalls both wheels if the imported `cv2` differs from the pin; `cloud_videos.py doctor` reports a mismatch | every frame decode |
| `torch`, `torchvision`, `easyocr` | OCR (CRAFT detector + recognizer). EasyOCR downloads its `en` weights to `~/.EasyOCR` on first run, so the machine needs outbound internet once | pip | `scripts/setup.*`; on Linux torch comes from the pytorch.org index matching the driver's CUDA version (or `$TORCH_INDEX_URL`) | `surgery_analysis_process.py` |
| `h5py`, `tqdm`, `scikit-learn`, `matplotlib` | frame cache, progress, tooling, GUI | pip | `scripts/setup.*` | `extract_roi_images.py`, `video_annotator_gui.py` |
| `psutil` (pinned 5.9.8) | CPU load sampling for idle-time jobs | pip | `scripts/setup.*` | `tools/backfill_regions.py --when-idle` |
| CRNN weights | fallback simple OCR model | `models/OCR_interface/simpleocr/*.pth` (tracked in git) | `git clone` | `models/OCR_interface` |
| Pattern / header templates | stage and header matching | `data/roi_img_caches/` (tracked in git) | `git clone` | `stage_pattern_analysis.py` |
| Google Cloud SDK (`gcloud`) | move videos and results through GCS | on PATH, or `tools/google-cloud-sdk/` (gitignored) | `scripts/setup.sh` downloads it on Linux; `winget install Google.CloudSDK` on Windows | `tools/cloud_videos.py` |
| `libgl1`, `libglib2.0-0` | native libraries OpenCV links against on Linux | apt | `scripts/setup.sh` | `cv2` import |
| `python3.X-venv` | `ensurepip`, which Debian/Ubuntu split out of the base Python | apt | `scripts/setup.sh`, only when missing | creating `.venv` |
| NVIDIA driver + CUDA-capable GPU | OCR speed. Optional: the pipeline falls back to CPU | host | cloud provider / you | `torch.cuda.is_available()`; `scripts/setup.sh` fails if `nvidia-smi` sees a GPU but torch cannot |

## Cloud storage layout

Bucket and prefixes are set in `config/cloud.json` (override with `--bucket` or
`SURGERYOCR_GCS_BUCKET`).

```
gs://xensur-surgeryocr-videos/
├── videos/<path relative to data/>        e.g. videos/0816/2025-08-16_135759.MP4
└── results/<video key without extension>/ e.g. results/0816/2025-08-16_135759/stage_analysis.json
```

`pull` recreates the `data/` layout, so `tools/golden_videos.json` paths resolve
unchanged on the cloud machine. Results are the `*.json` / `*.jsonl` files that
`run_pipeline.py` writes to `data/<video stem>/`.

## Authentication

The tool uses whatever account `gcloud` is logged in as. On a headless machine:

```bash
gcloud auth login --no-browser
```

It prints a `gcloud auth login --remote-bootstrap="..."` command. Run that on a
machine with a browser, paste the resulting URL back, done. The account needs
`roles/storage.objectAdmin` on the bucket (the bucket owner's account has it).

### Run the repository checks

After backend setup, install the development tools with the project interpreter
(`python -m pip install ruff==0.15.11 pytest==9.0.3` in the activated environment).
Run `./scripts/check.ps1` in PowerShell or `bash scripts/check.sh` on Linux.
The checks cover formatting, lint, and all automated tests under `tests/`.
For GUI tests on a headless Linux host, use `xvfb-run -a bash scripts/check.sh`.
