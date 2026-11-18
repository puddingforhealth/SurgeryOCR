# SurgeryOCR backend

Unzip the backend into a writable directory and run these commands from its root.
Install Python 3.11 or newer. Setup creates a project-local backend environment.
The lightweight viewer packaging environment is named `release_env`.

Windows PowerShell:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
python run_pipeline.py --video VIDEO.mp4 --output-dir results
```

Linux:

```sh
SKIP_GCLOUD=1 bash scripts/setup.sh
source .venv/bin/activate
python run_pipeline.py --video VIDEO.mp4 --output-dir results
```

Setup checks the OCR weights and imports. For a later check run
`python tools/install_ocr_bundle.py --doctor` in the backend environment.
The first EasyOCR initialization on a fresh host downloads its stock reader and
CRAFT assets for BSS; provision the EasyOCR cache first on offline hosts.

The CLI defaults to RAM mode and verifies the packaged model hashes. Automatic
machine detection selects fine-tuned EasyOCR for machine 1/2 numeric regions
1–6, with the original BSS reader alone in region7. Machine 3 uses eight RGB
fields and a dedicated numeric CTC model. The CPU backend uses ONNX Runtime;
CUDA uses TorchScript. Confirm the selected machine, route and bundle version
in the log and in `run_stats.json`. Use `--machine-id 1|2|3` for a known layout
when the identification header is absent. Use `--help` for all arguments.
Keep the output directory separate from earlier analyses.

## Output contract for timeline consumers

- Read `results/<video stem>/stage_analysis.json`. Existing fields include
  `video`, `total_frames` and `regions`; individual STAGE/PEDAL segments use
  inclusive `start_frame` and `end_frame`. Additional fields include
  `machine_id`, `regions.STATUS` (Occlusion, machines 1/2) and `regions.AS`
  (machine 2). A supported region can be an empty list.
- Read each `<region>_ocr_testing.jsonl` line independently. Single readings
  use `frame`; groups use `source_frame` and `matched_frames`, and records
  retain `type`, `ocr_text`, `confidence` and `setting`. Clear a field on
  `deactivate`. Hold its value between changes. Region names on machines 1/2
  are region1 through region7; machine 3 uses instrument field names.
- On machine 3, a consecutive equal run may have only its first frame in
  `matched_frames`. Optional `last_observed_frame` and `observed_frame_count`
  describe its extent. Readers should tolerate additional keys and use frame
  indices rather than assuming every observation has an explicit row.
- `frame_clock.json` maps indices to playback seconds. A service with verified
  per-frame `pts_sec` can use that directly. `analysis_layout.json`,
  `ocr_review.jsonl` (when produced) and `run_stats.json` are sidecars.
- This package writes analysis files. The HTTP service that constructs
  `/videos/{id}/metadata` owns its field names, serialization and compression.
  Pass AS and STATUS through the existing stage track representation. Keep
  change records sparse when the consuming API supports them.

The model bundle manifest identifies weights with SHA-256. It records upstream
EasyOCR and PaddleOCR/RapidOCR attribution. Training videos and review datasets
are stored separately. The backend distribution starts with an empty mapping
of video filenames to machine types.
