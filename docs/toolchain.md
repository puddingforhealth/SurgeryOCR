# Toolchain

Everything the pipeline needs, where it lives, and which step uses it.
`python tools/cloud_videos.py doctor` checks all of it and reports what is missing.

| Tool | Purpose | Where | Installed by | Used by |
|---|---|---|---|---|
| Python 3.10+ | runtime | Linux: `.venv/` (gitignored; a symlink when `VENV_DIR` is set). Windows: current Python | Linux: `scripts/setup.sh` creates the venv from `python3` (or `$PYTHON`); Windows: you | everything |
| `opencv-python`, `numpy`, `Pillow` | frame decode, binarize, template matching | pip | `scripts/setup.*` via `requirements.txt` | `run_pipeline.py`, `stage_pattern_analysis.py` |
| OpenCV version pin | `opencv-python` and `opencv-python-headless` (pulled in by EasyOCR) are both pinned to `4.11.0.86`. Both wheels ship the same `cv2/` package, and OpenCV 5.0 decodes the same video to different pixels (every frame, ~3 grey levels on average), which changes some OCR readings. Measured on 2026-10-20: with 4.11, a Linux H100 run matches the Windows run reading for reading. Upgrade only together with new golden baselines | `requirements.txt` | `scripts/setup.sh` reinstalls both wheels if the imported `cv2` differs from the pin; `cloud_videos.py doctor` reports a mismatch | every frame decode |
| `torch`, `torchvision`, `easyocr` | OCR (CRAFT detector + recognizer). EasyOCR downloads its `en` weights to `~/.EasyOCR` on first run, so the machine needs outbound internet once | pip | `scripts/setup.*`; on Linux torch comes from the pytorch.org index matching the driver's CUDA version (or `$TORCH_INDEX_URL`) | `surgery_analysis_process.py` |
| `h5py`, `tqdm`, `scikit-learn`, `matplotlib` | frame cache, progress, tooling, GUI | pip | `scripts/setup.*` | `extract_roi_images.py`, `video_annotator_gui.py` |
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
