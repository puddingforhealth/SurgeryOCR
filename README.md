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

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt

## Usage

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
| `surgery_stage_rois.json` | pattern-region boxes (STAGE / PEDAL) |
| `pattern_name_mapping.json` | pattern id -> display name |
| `diff_rule.json` | per-region matching rules; `"binarize": "rule"` matches roi_picker templates, `"pedal_rgb": false` disables the machine-1 RGB pedal path |
| `ocr_strategy.json` | OCR strategy: `ocr_input` (binary/gray), `stable_frames` debounce, `min_ocr_height` upscale, `diff_threshold` |

The pipeline resolves the route after machine detection (`utils/get_configs.py: get_machine_key / resolve_machine_config_path`), then runs stage analysis + OCR with that route's assets. Machine 3 uses `ocr_input: gray` + debounce because its translucent overlay contaminates binarized crops when the surgical background moves.

### Annotation-to-config workflow for a new machine

1. Annotate with `roi_picker.py` quick-capture (anchor, pattern templates, headers, value coords) — everything is recorded in `data/roi_img_caches/machine_<id>/capture_log.json`.
2. Compile into formal configs:
   ```bash
   python compile_machine_config.py --machine 3 [--dry-run]
   ```
   This writes `config/machine_<id>/…`, merges region coords into `config/rois.json`, and adds OCR char sets. Rerunnable; it also reports unlabeled patterns and missing assets.
3. Add a detection anchor entry to `config/machine_identity.json`.
