# SurgeryOCR



## Description

This project aims to automate the extraction and labeling of critical numerical data displayed within ophthalmological surgery videos. By leveraging CV+OCR(Optical Character Recognition) techniques, the repository provides tools to:

* Identify and extract numerical values appearing on the video feed.
* Label these extracted numbers based on the specific stage of the surgery they correspond to.
* Provide a structured dataset for further analysis of surgical procedures, potentially leading to insights for research, training, and quality improvement in ophthalmology.

## Key Features

* **Video Analysis:** Processes ophthalmological surgery videos as input.
* **On-Screen OCR:** Employs OCR to recognize numerical data displayed on the video frames.
* **Stage Identification:** Aims to identify different stages of the surgery to provide context to the extracted data.
* **Data Labeling:** Labels the extracted numerical data according to the identified surgical stage
* **Output:** Generates structured output (e.g., jsonl) containing the extracted numerical data and their corresponding labels and timestamps.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd SurgeryOCR
    ```

2.  **Install dependencies** (idempotent; re-run after every `git pull`):
    ```bash
    bash scripts/setup.sh          # Linux: builds .venv with torch, EasyOCR and requirements.txt
    source .venv/bin/activate      # in every new shell
    ```
    On Windows, run `.\scripts\setup.ps1` with Python 3.11+ available, then `.\.venv\Scripts\Activate.ps1`. The script installs and checks the backend in `.venv`. Use `-Python PATH` to select the interpreter. Use `-ReuseBasePackages` only with a base interpreter that already has the backend dependencies. Windows setup supports `TORCH_INDEX_URL` for a custom PyTorch wheel index. The lightweight `release_env` is for viewer packaging.

    You can also run the backend directly from the repository root in PowerShell:

    ```powershell
    .\.venv\Scripts\python.exe run_pipeline.py --video "path\to\video.mp4"
    ```

    To repeat the setup check, run `.\.venv\Scripts\python.exe tools/install_ocr_bundle.py --doctor`. This uses the project's Python, checks that the backend packages load, verifies the model file hashes, and reports CUDA availability. Setup already runs this check once. You can go straight to `run_pipeline.py` for subsequent videos.

    `scripts/setup.sh` also installs the system libraries OpenCV needs, the Google Cloud
    SDK, and stops with an error if `nvidia-smi` sees a GPU that torch cannot use.
    OpenCV is pinned to 4.11.0.86 (both `opencv-python` and EasyOCR's `opencv-python-headless`)
    because OpenCV 5.0 decodes the videos to different pixels and changes some OCR readings;
    setup verifies the `cv2` that actually imports and repairs a mismatch.
    On a GPU machine it reads the CUDA version the driver supports from `nvidia-smi` and
    installs torch from the newest pytorch.org CUDA index not above it (PyPI's default
    build targets the newest CUDA, which older drivers reject); a torch already built for
    a newer CUDA than the driver is replaced. Environment overrides:

    | Variable | Effect |
    |---|---|
    | `PYTHON=python3.11` | interpreter the venv is created from (default `python3`) |
    | `VENV_DIR=/big/disk/venv` | keep the venv on another disk; `.venv` becomes a symlink to it, so `source .venv/bin/activate` still works |
    | `TORCH_INDEX_URL=<url>` | pip index for torch, overriding the automatic choice (e.g. `https://download.pytorch.org/whl/cu126`) |
    | `SKIP_GCLOUD=1` | skip the Google Cloud SDK when videos arrive another way (e.g. `rcs`) |

## Usage

### Single-pass pipeline (`run_pipeline.py`)

One command decodes each video once and writes OCR and stage results:

```bash
python run_pipeline.py --video data                     # every .mp4 under data/, subfolders included
python run_pipeline.py --video data/0816/clip.MP4       # one video
python run_pipeline.py --video data --output-dir runs/cloud
```

| Flag | Default | Meaning |
|---|---|---|
| `--video PATH` | required | a video file, or a directory searched recursively for `.mp4` (any letter case) |
| `--output-dir DIR` | `data` | results go to `DIR/<video stem>/` (`stage_analysis.json`, `*_ocr_testing.jsonl`, `run_stats.json`, `frame_clock.json`) |
| `--mode` | `ram` | `ram` keeps frames in memory (fastest); `frame` saves full frames; `detail` saves frames and ROI crops; `read` reuses a saved frame cache |
| `--force` | off | overwrite an existing frame cache and ROI images |
| `--ocr-profile` | `auto` | `auto` uses installed verified weights, `legacy` uses original OCR, `enhanced` requires the trained bundle |
| `--ocr-bundle PATH` | local installed bundle | override bundle location (or `SURGERYOCR_OCR_BUNDLE`) |
| `--ocr-device` | `auto` | `auto`, `cpu` or `cuda`; CUDA scene inference needs the packaged TorchScript model |
| `--ocr-sensitivity` | `standard` | original machine-1/2 trigger threshold, or `exact` for any binary pixel change; machine 3 always checks RGB each frame |
| `--machine-id 1\|2\|3` | mapping / auto detection | explicitly use a known layout without editing the video mapping |

Because results are keyed by file stem, a directory run refuses to start when two
videos in different subfolders share a name; rename one or run them separately.
A failure on one video is reported and the batch continues; the exit code is 1 if any failed.

### Step-by-step tools (legacy cache workflow)

Use the single-pass pipeline above for integrated trained OCR profiles. The
standalone cache tools below retain their original OCR path and are useful for
legacy inspection; installing a bundle does not reinterpret old binary caches
as machine-3 RGB inputs.

This is the standard operating procedure for analyzing surgery videos. These commands assume you are running them from the root of the project directory.

1.  **Place Videos**: Put all surgery videos (e.g., `.mp4` files) into the `data/` directory.

2.  **Extract ROI Images**: Pre-process the videos to extract ROI images. This step significantly speeds up the subsequent analysis and UI responsiveness. The following command processes all videos in the `data/` directory and extracts all configured ROIs, including saving full frames for faster UI loading.
    ```bash
    python extract_roi_images.py --video ./data --region all --save-full-frames
    ```

3.  **Run OCR Analysis**: Perform change detection and OCR on the extracted ROIs. This command processes all videos and generates `.jsonl` files with the OCR results.
    ```bash
    python surgery_analysis_process.py --video ./data --region all
    ```

4.  **Analyze Surgery Stages**: Run pattern matching to detect and segment different surgical stages based on the visual patterns in the ROIs.
    ```bash
    python stage_pattern_analysis.py --video ./data
    ```

5.  **Review Results**: Launch the graphical user interface to load a video and visually inspect the analysis results, including OCR data and surgical stage segments.
    ```bash
    python video_annotator_gui.py
    ```
    For each video the GUI loads results from `<video folder>/<stem>/`, else the first
    `data/**/<stem>/` holding analysis files, else `data/<stem>/`. To review results kept
    elsewhere, e.g. pulled from the cloud into `runs/cloud/`, click **分析資料夾…** and pick
    that root: `<root>/<stem>/` is then preferred and the open video reloads from it. The label
    beside the button shows the folder in use. Setting `ANALYSIS_DATA_ROOT=runs/cloud` before
    launching does the same. Annotations you save go to the folder in use.

    The video automatically fits the available viewing area while keeping its aspect ratio:
    either its width or height fills the area, with no cropping or stretching. Resize or
    maximize the window to enlarge it. ROI overlays scale with the image, and resizing a
    paused video keeps the same frame.

    The dark workstation layout groups video/file actions, the two pages (**數值追蹤**, the OCR
    value readings, and **狀態追蹤**, the pattern-matched states), ROI frame comparisons, the
    event table, and playback/timeline controls. The status footer shows current actions and
    keyboard shortcuts: **Space** plays/pauses, **Left/Right** step frames, and **Up/Down** move
    between events. Both pages and annotation files keep their existing behavior. The window
    supports sizes from 1050×720 upward.

    The timeline card has the play button, frame and time readouts, and frame jump on top,
    then one labelled row per axis, all on one x scale. The first row, 幀軸 FRAME, is the
    whole-video seek axis with minute ticks. The analysis tracks follow: 踏板 PEDAL, 階段 STAGE,
    and, when the analysis has them, AS 圖示 and Occlusion. The arrow keys drive the panel you
    clicked last, and the hint strip in the footer shows which panel that is and what each key
    does there. Clicking the frame axis seeks, and then **Left/Right** step single frames
    (**Shift** about a second). Clicking a track (or its label) makes **Left/Right** jump to the
    previous/next segment. The segment under the playhead is framed in white, and the footer
    shows "segment n/N" with its times. **Up/Down** move between the frame axis and the tracks;
    a track also lists its events on the right. **Esc** or a click on the event list gives the
    arrow keys back to the list.

    The ROI frame comparison (last / current frame) sits in the top band on the right, beside
    the file row and page tabs, and is exactly as wide as the event list under it. The previews
    are scaled to fit that block, and the event list runs the full height of the video card.
    The event list explains that a double-click or **Enter** corrects an annotation, and that a
    right-click deletes one or shows details. Stepping through the list scrolls it only when the
    selected row would leave the visible area, one row at a time.
    See [workstation layout and verification](docs/viewer-workstation.md) for details.
    **F1** opens the complete shortcut reference. **Ctrl+O** opens a video, **Ctrl+S** saves
    OCR annotations, **Ctrl+L** focuses the frame jump field, **Alt+1/2** switches between the two pages (the tabs show it),
    and **Alt+Up/Down** switches ROI regions. Manual frame/time changes briefly highlight
    the readouts; this feedback can be disabled in the F1 panel.
    Event counts, alternating rows, and source resolution/FPS help orient the review.
    Hover over filenames or result paths to see them in full. Buttons accept **Enter**;
    annotation dialogs accept **Esc** to cancel. Invalid frame jumps show inline guidance,
    and shortcut saves report success in the status footer.
    Frame navigation and displayed times use the video's actual presentation timestamps;
    the slider and both timeline tracks share that time axis. See
    [frame/time synchronization](docs/viewer-timing.md) for VFR verification details.

## Running on a Cloud Machine

For Google Drive inputs, see [Drive batch analysis](docs/drive-batch.md): download one video at a time
directly on the GPU machine while analyzing the previous one in RAM mode, then verify and collect
each result locally with the original folder hierarchy. Temporary inputs are bounded and successful
copies are removed only after local collection is acknowledged.
For local review, the same workflow can download videos serially with a bandwidth limit and
place each MP4 beside its matching analysis folder on a data drive; the GUI finds those results automatically.

### Fetching results through `rcs`

`rcs pull` copies one file, so pack the result folder first (pax keeps UTF-8 file names)
and unpack it with Python, which reads those names correctly on Windows:

```powershell
rcs 'cd /root/SurgeryOCR && tar --format=pax -czf /tmp/cloud-results.tgz -C runs cloud'
rcs pull /tmp/cloud-results.tgz build/cloud-results.tgz
python -X utf8 -m tarfile --filter data -e build/cloud-results.tgz runs
```

Results land in `runs/cloud/<stem>/`; open the GUI, click **分析資料夾…** and choose `runs/cloud`.
Re-running the three commands refreshes them.

### Through a GCS bucket

Videos are too large for git, so they travel through a GCS bucket
(`gs://xensur-surgeryocr-videos`, set in `config/cloud.json`). One setup script
installs everything, including the Google Cloud SDK if it is missing:

```bash
git clone <repo> && cd SurgeryOCR
bash scripts/setup.sh                      # Linux (cloud); scripts/setup.ps1 on Windows
source .venv/bin/activate                  # every new shell
gcloud auth login --no-browser             # once per machine; paste the printed command locally
python tools/cloud_videos.py doctor        # packages, GPU, templates, bucket access
python tools/cloud_videos.py pull --all --run --upload-results
```

The last line downloads every video into `data/`, runs `run_pipeline.py --mode ram`
on each, and uploads `data/<video>/*.json*` back to the bucket. Back on the
workstation, `python tools/cloud_videos.py pull-results --all` fetches them.

### `tools/cloud_videos.py`

| Command | What it does |
|---|---|
| `doctor` | Check Python packages, CUDA, tracked weights/templates, `gcloud` login and bucket access |
| `list` | Show the videos and result directories in the bucket with sizes |
| `push <paths...> [--force]` | Upload video files or directories. Keys mirror the path under `data/`; same-size objects are skipped |
| `pull [names...] [--all] [--force] [--run] [--mode M] [--upload-results]` | Download by name, stem, key or glob. `--run` analyses each video after download; `--upload-results` pushes its outputs |
| `push-results [names...] [--all] [--force]` | Upload `data/<video>/*.json*` for the given videos (`--all`: every video that has a result directory) |
| `pull-results [names...] [--all]` | Download result directories into `data/` |

Global option `--bucket gs://...` (or `SURGERYOCR_GCS_BUCKET`) overrides the
configured bucket. Transfers use `gcloud storage`, so multi-GB uploads resume on
retry. Bucket layout and authentication details are in
[`docs/toolchain.md`](docs/toolchain.md).

## Golden-Regression Testing

> Measured timings, and the optimisations that were tried and rejected, are recorded
> in [`docs/pipeline-performance.md`](docs/pipeline-performance.md). Read it before
> attempting performance work — a raw-pixel change gate, FFmpeg decode tuning and
> skipping the OCR text detector have all been measured and ruled out (or limited
> to specific machines).

Most pipeline changes are meant to be behaviour-preserving (performance work,
refactoring). Those are verified by running a fixed set of videos and diffing the
outputs against recorded reference outputs — the pipeline is deterministic, so a
correct change reproduces the baseline byte for byte.

```bash
# record reference outputs (do this on a known-good commit)
python tools/golden.py baseline --set full

# after a change: run again and diff against the baseline
python tools/golden.py check --set fast     # 2 short videos, ~1 min
python tools/golden.py check --set full     # 3 machine types, ~18 min
```

| Flag | Meaning |
|---|---|
| `baseline` / `check` | record reference outputs, or run and compare against them |
| `--set fast\|full` | which video set from `tools/golden_videos.json` to run |
| `--python PATH` | interpreter that has torch + easyocr (default: the current one) |
| `--rtol FLOAT` | relative tolerance for `avg_rmse` / `confidence` (default `1e-6`) |

Outputs and wall-clock timings land in `runs/` (gitignored):
`runs/baseline/<video>/`, `runs/candidate/<video>/`, plus a `timing.json` in each.
`check` prints the wall-clock delta against the baseline, so a change can be judged
on both correctness and speed in one command.

The test set in `tools/golden_videos.json` covers all three machine types and both
pattern-matching paths (the RGB pedal icon on machines 1/2, the rule-binarized digit
on machine 3). Add a video there when a new machine type or code path appears.

> Wall-clock on the short videos varies noticeably with the OS file cache — a cold
> first read is slower than a repeat. Compare timings from the `full` set, or repeat
> a run, before believing a small delta.

### Where the time went: `run_stats.json`

Every run writes `run_stats.json` next to `stage_analysis.json`:

```json
{ "frames": 2367, "ms_per_frame": 6.031,
  "timings": { "decode_wait_seconds": 7.31, "stage_seconds": 1.22, "ocr_seconds": 4.41 },
  "work": { "template_compares": 59175, "ocr_calls": 98, "changes_detected": 79 },
  "per_region": { "region1": { "change_count": 19, "ocr_count": 17, "ocr_seconds": 0.59 } } }
```

`decode_wait_seconds` is the time the main loop sat blocked waiting for a frame —
decoding runs on a background thread, so this is the part of it that analysis could
not hide, i.e. the decode cost actually paid.

The `work` counters matter as much as the timings. Timings say whether a change got
faster; the counters say whether it got faster by doing the same work more
efficiently or by quietly skipping some of it. A performance change is only valid
when the outputs match the baseline **and** the counters move the way the change
predicts. `template_compares` is the one to watch when adding pattern regions: it
grows as regions x templates x frames.

### Frontend compatibility

For a machine-1/2 single-recognizer bundle, pass its directory with
`python run_pipeline.py --video VIDEO.mp4 --mode ram --ocr-profile enhanced --ocr-bundle BUNDLE_DIR`.
Package these weights with `tools/build_ocr_bundle.py --machine12-route single_numeric`
and the existing required build arguments. The default build route is `conservative`.
The version 2 manifest setting `"machine12_route": "single_numeric"` selects the fine-tuned
recognizer for regions 1–6 and the original reader for BSS. Each region uses one
model. A blank may trigger the existing retry on a border-trimmed crop.
The startup line and `run_stats.json` identify this backend as
`easyocr-single-numeric`. This setting preserves the region JSONL and stage
interval formats. Machine 3 uses its existing RGB model.
See the [single-model training and confirmation results](docs/ocr-single-model-trial.md)
for the trial bundle, remaining errors and comparison against the dual route.

The January pipeline (`a05e1f0`) and October main (`91cf542`) both write
`<region>_ocr_testing.jsonl` and `stage_analysis.json`. Readers should handle
individual OCR records and grouped records with `source_frame` and
`matched_frames`. Region coverage depends on the machine and analysis version.

The metadata service consumed by the web frontend exposes a separate HTTP
representation. Preserve its per-frame `pts_sec`, `ocr.regionN`, `fields`,
`role`, `status`, `provenance`, and `stage_regions` when adding events.
The source for Occlusion is `stage_analysis.json → regions.STATUS`
(`pattern_name == "Occlusion"`), and the source for Active Sentry is
`regions.AS`. Both segment endpoints are inclusive decoded frame indices.
Use the existing stage event representation for these additional tracks.
Machine 1 supports STATUS, machine 2 supports both, and machine 3 supports
neither. An empty supported track means the analysis found no events. A missing
supported track indicates incomplete or older data and should be surfaced.

For playback, an event remains active through its end frame. When another frame
follows, the exclusive end time is the presentation time of that next frame.
HTTP gzip preserves the existing per-frame payload. Removing carried-forward
rows changes the response contract and requires a separately negotiated format.

### Frame index to playback time: `frame_clock.json`

OCR and stage results identify a frame by its decode index. A player seeks by
presentation time. The two agree as `index / 30` only while the recording runs
smoothly. Some recorders stall: one frame is held for seconds and the pictures that
follow cannot be decoded, yet they still advance the clock. Of 117 recordings
scanned, 7 stall 1.5 s after the start (every later frame runs about 2–3 s late)
and 2 stall mid-surgery. A web player that converts `index / 30` then shows the
wrong moment from the stall on.

Every decoding run (`--mode ram|frame|detail`) writes `frame_clock.json` next to the
OCR results:

```json
{
  "version": 1,
  "source": "2024-10-04李慧娜os.MP4",
  "nominal_fps": 30.000579,
  "decoded_frames": 30512,
  "dropped_frames": 15,
  "segments": [
    { "frame": 0,  "time": 0.0 },
    { "frame": 46, "time": 3.989633 },
    { "frame": 47, "time": 4.523 }
  ]
}
```

To find the time of frame `f`, take the last segment with `frame <= f`:

```
time(f) = segment.time + (f - segment.frame) / nominal_fps
```

A recording without stalls has the single segment `{ "frame": 0, "time": 0.0 }`.
Times are seconds from the start of the video stream, matching what a browser
`<video>` reports as `currentTime`. The table is exact to within half a frame. To
show frame `f`, seek to `time(f) + 0.5 / nominal_fps` so rounding cannot land on the
previous frame. `dropped_frames` counts pictures the container lists but that never
decode (`null` when the container gives no frame count). `read` mode decodes no
video, so it leaves an existing clock untouched.

The times are read from the same OpenCV decode that produced the OCR indices
(`CAP_PROP_POS_MSEC` after each `read()`). On the recording above they match the
PyAV presentation timestamps on all 30,512 frames. The format is the one
PhacoSense's own `frame_clock.py` scanner writes, so one reader handles both.

Results produced before this file existed can be completed without rerunning OCR.
The tool decodes each video once (about 5 minutes for a 17-minute 1080p video):

```bash
python tools/frame_clock.py --video D:/batch --jobs 4            # results folder beside each video
python tools/frame_clock.py --video data/0816 --output-dir data  # results in data/<video stem>/
```

| Flag | Default | Meaning |
|---|---|---|
| `--video PATH` | required | a video file, or a directory searched recursively for `.mp4` |
| `--output-dir DIR` | beside the video | results root used with `run_pipeline.py`; writes `DIR/<video stem>/frame_clock.json`. Omitted: the folder next to the video with its name (the layout `tools/drive_batch.py` collects into) |
| `--force` | off | rewrite an existing `frame_clock.json` |
| `--jobs N` | `1` | videos decoded in parallel |

Each video prints `[ok]`, `[skip]` (no results folder yet, or the clock already
exists), or `[fail]`. The tool refuses to write a clock when `run_stats.json` from
the OCR run reports a different decoded frame count. That would mean this decode is
not the one the OCR indices came from. The exit code is 1 if any video failed.

The pure functions live in `utils/frame_clock.py`: `build_clock(source, frames,
container_frames)`, `frame_time(clock, frame)`, `validate_clock(clock)`,
`read_clock(path)`, `write_clock(path, clock)` and `scan_video(path)`.

### Comparing two runs directly

```bash
python tools/compare_analysis.py <baseline_dir> <candidate_dir> [--rtol 1e-6] [--quiet]
```

Structural fields (`pattern`, `pattern_name`, `start_frame`, `end_frame`,
`frame_count`, `ocr_text`, `setting`, `matched_frames`) are compared exactly;
`avg_rmse` and `confidence` use the relative tolerance, because cuDNN kernels are
not guaranteed to be bit-reproducible across machines. Exit code 0 means the runs
are equivalent. Use `--rtol` with a larger value when a change is expected to
perturb scores but not the segmentation (for example when adding a change-detection
gate that reuses a previous frame's score).

## Development Tools

### ROI Picker (`roi_picker.py`)

Interactive tool for grabbing pixel coordinates and binarized templates from videos or screenshots. Useful when annotating a new machine layout (e.g. locating labels such as "CDE", stage text, or value regions).

```bash
python roi_picker.py                        # pick a file via dialog
python roi_picker.py --video data/xxx.mp4   # open a video
python roi_picker.py --video shot.png       # open a single image
```

| Control | Action |
|---|---|
| Mouse drag | Draw the ROI rectangle |
| `Left` / `Right` | Previous / next frame (`Shift`: 10, `Ctrl`: 100) |
| `W`/`A`/`S`/`D` | Nudge the rectangle by 1 px (`Shift`: 5 px) |
| Coordinate fields | Type exact values, then "套用座標" |

Features: live binarization preview with adjustable HSV-saturation / gray thresholds, white-pixel-ratio readout, copy `[x1, y1, x2, y2]` to clipboard, and saving binarized / raw crops (for pattern or anchor template caches) or the full frame as PNG.

**Quick capture mode** (機型樣板快速存檔): given a machine ID, saves the current selection to the conventional cache layout without any dialogs, and records every coordinate in `data/roi_img_caches/machine_<id>/capture_log.json` (including the binarize thresholds used):

* `Pattern樣板` — auto-numbered `machine_<id>/<區域名>/<n>.png` + `.npy` (e.g. STAGE, PEDAL); enforces one shared box per pattern region (warns when the selection drifts from the recorded coords).
* `Header樣板` — `machine_<id>/roi_headers/<region>.png` + header coords, keyed by the 區域名 field.
* `數值區座標` — coordinates only (the changing value area OCR will read); no image saved.

The capture log is the staging area from which region entries are later compiled into `config/rois.json`, `config/surgery_stage_rois.json`, and `config/pattern_name_mapping.json`.


## Machine Type Detection

The pipeline supports multiple machine screen layouts ("machine types"). Detection is anchor-based and configured in `config/machine_identity.json`:

* Each machine type declares one **anchor**: a binarized template of a UI element at a fixed screen position (e.g. the "CDE" label for the UNITY machine, the region1 header for machines 1/2).
* Every frame, each anchor is compared via XOR diff ratio against the binarized crop at its coordinates. The first anchor with `diff < diff_threshold` casts a vote; a machine type is confirmed after `consecutive_votes` identical votes in a row.
* Until confirmation, frames are buffered (JPEG-encoded) and retroactively OCR-processed once the type is known. If the buffer exceeds `max_buffer_frames` — or the video ends unresolved — `fallback_machine_id` is used.
* Videos already present in `video_machine_mapping` (in `config/rois.json`) skip detection entirely.

Pattern/template caches are organized per screen-layout route under `data/roi_img_caches/`:

```
data/roi_img_caches/
├── machine_1_2/            # machines 1 & 2 (same layout, offset coordinates)
│   ├── STAGE/ PEDAL/ ...   # stage pattern templates (<pattern_id>.npy + .png)
│   └── roi_headers/        # per-region header (label) templates
└── machine_3/              # UNITY machine (completely different layout)
    ├── anchor.png          # machine-type detection anchor ("CDE" label)
    ├── STAGE/              # stage pattern templates
    └── roi_headers/        # per-region header templates
```

To add a new machine type: grab its anchor template and coordinates with `roi_picker.py`, save the binarized crop under `data/roi_img_caches/machine_<id>/`, and append an entry to the `anchors` list.


## Multi-Machine Support (Pattern Routes)

Two screen-layout "pattern routes" are supported: `machine_1_2` (machines 1 & 2, opaque panels) and `machine_3` (UNITY, translucent overlay). Machine-specific configs live in `config/machine_<key>/` and override the legacy shared files; anything absent falls back to `config/`:

| File | Purpose |
|---|---|
| `surgery_stage_rois.json` | pattern-region boxes (STAGE / PEDAL / AS / STATUS); a box can differ per machine, see below |
| `pattern_name_mapping.json` | pattern id -> display name |
| `diff_rule.json` | per-region matching rules; `"binarize": "rule"` matches roi_picker templates, `"binarize": "hsv_mask"` with `hsv_ranges` keeps only pixels of the listed colours, `"pedal_rgb": false` disables the machine-1 RGB pedal path |
| `ocr_strategy.json` | OCR strategy: `ocr_input` (binary/gray), `stable_frames` debounce, `min_ocr_height` upscale, `diff_threshold` |

The pipeline resolves the route after machine detection (`utils/get_configs.py: get_machine_key / resolve_machine_config_path`), then runs stage analysis + OCR with that route's assets. Machine 3 uses `ocr_input: gray` + debounce because its translucent overlay contaminates binarized crops when the surgical background moves.

### Boxes that differ between machines 1 and 2

Machines 1 and 2 share one pattern route, but not every region sits at the same
place on both screens, and some regions exist on only one of them. Such a region
lists a box per machine id; a machine without an entry does not get the region:

```json
"AS":     {"by_machine": {"2": [1738, 542, 1838, 640]}},
"STATUS": {"by_machine": {"1": [1628, 682, 1836, 718], "2": [1628, 664, 1836, 700]}}
```

`read_surgery_stage_rois(path, machine_id)` resolves the file for one machine,
and `update_stage_roi(raw, region, box, machine_id)` writes an edited box back
without flattening the per-machine entries (the annotator GUI saves through it).

### Status regions: AS and STATUS

| Region | Machines | Patterns | What it is |
|---|---|---|---|
| `AS` | 2 | `1` AS | Active Sentry icon below the Vac panel. Machine 1 does not display it. |
| `STATUS` | 1, 2 | `1` Occlusion | Status message panel below Vac / AS. Only Occlusion is tracked; Vent shows in the same panel but has no template, so it never produces a segment |

Both panels are translucent, so the surgical scene shows through. Matching
therefore uses `"binarize": "hsv_mask"`: only the text colours are kept (white
for the AS glyph; red and near-black for the status text), and the result is
compared with the templates in `data/roi_img_caches/machine_1_2/{AS,STATUS}/`.
Thresholds in `config/diff_rule.json` come from sampled frames. For STATUS the
RMSE is at most 50 for the same message, at least 106 for the other message, and
at least 75 with no panel; the threshold is 62. For AS it is 10–15 when shown and
70 when absent; the threshold is 35.

A new message (another status text) is one template image plus one line in
`pattern_name_mapping.json`. Machine 3 (UNITY) shows neither panel.

`stage_analysis.json` records `"machine_id"`, so a reader can tell "this machine
does not display AS" (region absent) from "AS never appeared" (empty list).

### Adding new regions to finished analyses: `tools/backfill_regions.py`

Results produced before a region existed can get it without rerunning OCR.
Each video is decoded once. In that pass the tool confirms the machine type,
matches only the requested regions, and rebuilds `frame_clock.json`. It adds the
regions and `machine_id` to `stage_analysis.json`; STAGE, PEDAL and the OCR files
are not touched. Finished videos are skipped, so the command can be stopped and
rerun at any time.

```bash
# one video after another, at full speed
python tools/backfill_regions.py --video D:/batch

# only while the rest of the PC is idle; safe to leave running
python tools/backfill_regions.py --video D:/batch --when-idle --log D:/batch/_backfill.log
```

With `--when-idle`, each video runs in a child process at idle priority. A video
starts only after other programs have used less than `--start-below` percent of
the CPU for `--settle` seconds. It is killed as soon as other programs use more
than `--stop-above` percent for two samples in a row, and retried later from the
start. Each line of the log says which video started, finished, or was stopped
because the CPU was busy. Create the `--stop-file` to end the runner cleanly.

| Flag | Default | Meaning |
|---|---|---|
| `--video PATH` | required | video file, or folder searched recursively |
| `--output-dir DIR` | beside the video | results root used with `run_pipeline.py`; omitted means the folder next to each video |
| `--regions R ...` | `AS STATUS` | regions to add |
| `--force` | off | redo videos that already have the regions |
| `--when-idle` | off | run only while the PC is idle (see above) |
| `--start-below P` / `--stop-above P` | `25` / `40` | other programs' CPU use, in percent of the whole machine |
| `--settle S` / `--sample S` | `60` / `5` | seconds of quiet before starting / seconds between samples |
| `--stop-file PATH` | none | exit when this file exists |
| `--log PATH` | none | also append progress lines to this file |

The tool refuses to write when `run_stats.json` from the OCR run reports a
different decoded frame count, and when the machine type cannot be confirmed
(add the video to `video_machine_mapping` in `config/rois.json` instead of
guessing).

### Annotation-to-config workflow for a new machine

1. Annotate with `roi_picker.py` quick-capture (anchor, pattern templates, headers, value coords) — everything is recorded in `data/roi_img_caches/machine_<id>/capture_log.json`.
2. Compile into formal configs:
   ```bash
   python compile_machine_config.py --machine 3 [--dry-run]
   ```
   This writes `config/machine_<id>/…`, merges region coords into `config/rois.json`, and adds OCR char sets. Rerunnable; it also reports unlabeled patterns and missing assets.
3. Add a detection anchor entry to `config/machine_identity.json`.

## Reviewed numerical OCR training and model comparison

Reviewed OCR datasets and fine-tuning: [workflow and full CLI reference](docs/ocr-finetuning.md).
The tools validate image hashes and video-disjoint splits, fine-tune the EasyOCR
English gen2 recognizer, export compatible weights, and prepare an explicit rcs
job without submitting it. Private medical crops and experimental weights stay outside Git. The selected production bundle in `models/trained_ocr/` can ship with the pipeline. Verify it with `python tools/install_ocr_bundle.py --doctor`.

```powershell
python tools/ocr_finetune.py validate --data DATASET --accept-assistant-labels
python tools/ocr_finetune.py train --data DATASET --weights WEIGHTS/english_g2.pth --output OUTPUTS/smoke --device cpu --epochs 1 --max-steps 2 --accept-assistant-labels
python tools/prepare_ocr_job.py --data DATASET --weights WEIGHTS/english_g2.pth --output JOB_FOLDER --accept-assistant-labels
rcs job pack --file JOB_FOLDER/rcs-job.json
```

Replace uppercase paths with real paths. Validation prints split/sign/decimal
counts; the smoke test writes loss, validation metrics and checkpoints; job
preparation creates a local `rcs-job.json`. It does not run training remotely.
GPU training refuses existing compute workloads and uses a configurable allocator
budget; these checks do not provide hardware isolation on a shared GPU.

For machine 3, see the [measured low-cost model comparison](docs/ocr-model-evaluation.md).
An optional CPU-only comparison environment is installed with
`python scripts/setup_ocr_benchmark.py`; then run
`.venv-ocr-benchmark/Scripts/python tools/benchmark_ocr_onnx.py --data REVIEW_FOLDER --model LOCAL_MODEL.onnx --output NEW_RESULT_DIR`
(Linux/macOS: `.venv-ocr-benchmark/bin/python`). `--threads` defaults to 2;
`--data` accepts multiple review folders. All options and the input schema are
documented in the linked comparison. Full per-image and per-region results are
written to `evaluation.json`; images are never uploaded by this tool.

The [larger machine-1/2 experiment](docs/ocr-broad-validation.md) records six
local GPU trials, video-disjoint test results, raw-model regressions, the
optional conservative prediction policy, and separately preserved label errata.
Use `--desktop-gpu` explicitly for local Windows WDDM training; shared Linux
targets retain the strict idle-GPU policy by default.

```powershell
python tools/ocr_finetune.py train --data DATASET --weights WEIGHTS/english_g2.pth --output OUTPUTS/local-gpu --device cuda --desktop-gpu --max-vram-gib 2 --batch-size 16 --accept-assistant-labels
python tools/compare_ocr_models.py --data DATASET --weights CANDIDATE/numeric_m12.pth --output OUTPUTS/comparison --split val --device cuda --desktop-gpu --candidate-detector whole_roi --prediction-policy conservative
```

The comparison retains the original CRAFT call as baseline, reports both gains
and regressions, and records any detector/policy change explicitly. See the
linked experiment for all flags and the limits of assistant-reviewed labels.

Machine 3 also has a [dedicated RGB PP-OCRv6 training workflow](docs/ocr-machine3-experiments.md),
with Small/Tiny comparisons, video holdouts, explicit blank labels and a
real-only versus synthetic-background ablation. The preserved experiment is
separate from the [integrated production profiles](docs/ocr-production-integration.md).
Setup and a minimal training run:

```powershell
python scripts/setup_ocr_scene.py
python tools/fetch_scene_ocr_models.py --model v6-small --output build/scene-models
.venv-ocr-scene/Scripts/python tools/scene_ocr_experiment.py --data DATASET --architecture PP-OCRv6_rec_small --weights build/scene-models/v6-small.pth --dictionary build/scene-models/v6-small-dictionary.txt --training-video m3v1 --output build/scene-fit --device cuda --desktop-gpu --gpu-memory-gib 2 --accept-assistant-labels
```

Success prints verified artifact paths and epoch progress, then writes the
plan, selected checkpoint, held-out predictions and compact ONNX model.
Use `-h`/`--help` on either tool; the linked workflow documents every flag,
dataset field, helper signature, output and limitation. CPU mode is the default;
Linux/macOS use `.venv-ocr-scene/bin/python` and retain strict shared-GPU checks.

## Integrated old/new OCR pipelines

The checked-in backend defaults to bundle `2026.11.18.single.2`. Machine 1/2 uses
the fine-tuned recognizer for regions 1-6 and the original reader for BSS, one
recognizer per region. Machine 3 uses the dedicated RGB recognizer.
See the [backend handoff and output contract](docs/backend-handoff.md).

```powershell
python tools/install_ocr_bundle.py --doctor
python run_pipeline.py --video PATH_TO_VIDEO --output-dir runs/new-analysis
```

`--ocr-profile auto|legacy|enhanced` defaults to enhanced and requires the bundled weights; `--ocr-bundle PATH` overrides the local bundle;
`--ocr-device auto|cpu|cuda` selects the device. Explicit enhanced mode refuses
missing/corrupt artifacts. Success prints the route and writes values, review
records, saved layout and model/timing provenance. Enhanced CPU OCR requires
Python 3.11+ and pinned ONNX Runtime from the normal setup requirements.
Machine 1/2 defaults to `--ocr-sensitivity standard`, preserving the original
change-trigger threshold. `--ocr-sensitivity exact` detects any binary pixel
change at potentially much greater cost. Both use exact cache matching;
machine 3 always checks source RGB each decoded frame.

```powershell
python tools/compare_pipeline_runs.py run --plan PRIVATE_PLAN.json --output runs/pipeline-comparison --device cuda
python tools/compare_pipeline_runs.py report --plan PRIVATE_PLAN.json --runs runs/pipeline-comparison --output runs/pipeline-report --machine12-data M12_DATASET --machine3-data M3_DATASET
```

The report checks stage/time/frame equality and compares the value actually
available at each annotated frame, including trigger/cache/blank handling.
See [setup, training, packaging, every new CLI flag and API](docs/ocr-production-integration.md)
for `build_ocr_bundle.py`, `install_ocr_bundle.py`, `compare_pipeline_runs.py`
and `evaluate_ocr_bundle.py`, plus rollback and annotation/testing limitations.

The [expanded fixed-model validation results](docs/ocr-expanded-validation.md)
cover 4,313 eligible machine-1/2 field-frame labels and 12 complete paired video
runs. Machine-1/2 reports separate full-pipeline and stage-active matches,
challenge subsets and all 11 regressions; machine-3 development diagnostics have
a separate report. The expansion does not retrain the models or establish a
weight-only gain. Its evidence is separate from the earlier integration checkpoint.

The [machine-3 pixel-color study](docs/ocr-machine3-color-study.md) measures
bright, low-saturation digit strokes and overlapping background reflections.
It includes exploratory mask diagnostics, not a new OCR accuracy claim.
