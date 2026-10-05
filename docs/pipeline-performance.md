# Pipeline performance: what was measured

A log of measured results for the single-pass analysis pipeline, including the
approaches that were tried and rejected. Several plausible-sounding optimisations
turned out not to work; they are recorded here so they are not attempted again.

Hardware for all numbers below: RTX 3070, 16 logical cores, Windows 11, Anaconda
base environment (torch 2.8.0+cu126, easyocr 1.7.2, opencv 4.11.0). Videos are
1920x1080 H.264.

## Method

Wall-clock only — `cProfile` inflates Python-level call costs enough to mislead
(it attributed 61 s to `named_modules` alone), so it is useful for finding hot
spots and not for comparing before and after.

Correctness is checked with `tools/golden.py`, which diffs a run against recorded
reference outputs. The pipeline is deterministic: two runs of the same commit on
the same video produce **byte-identical** `stage_analysis.json` and
`*_ocr_testing.jsonl`, confidence floats included. So a behaviour-preserving change
is held to byte-identical output, not to "close enough".

## Where the time goes

From `run_stats.json` after the decode/analysis overlap landed:

| video | machine | frames | ms/frame | decode (unhidden) | OCR | stage |
|---|---:|---:|---:|---:|---:|---:|
| 1131009何素珠od | 1 | 26,836 | 11.17 | 18.8% | **70.1%** | 6.1% |
| 0450713odlisa | 3 | 42,656 | 7.02 | 27.8% | **57.0%** | 7.4% |
| 2025-08-16_133917 | 2 | 2,622 | 6.31 | **52.7%** | 27.6% | 9.8% |
| 2025-08-16_135759 | 2 | 2,367 | 6.36 | **50.6%** | 31.9% | 8.2% |

The split depends almost entirely on how many readings a video produces. Videos with
dense readings are OCR-bound; videos with sparse readings are decode-bound. On
`1131009何素珠od`, `region3` alone accounts for 120 s of a 300 s run — 4,618 OCR
calls at 26 ms each.

Software H.264 decode costs 6.5 ms/frame cold, ~4.9 ms warm. That is the floor for
any approach that decodes every frame on the CPU.

## What worked

**Overlapping decode with analysis** (`PrefetchIterator`). Decoding happens inside
FFmpeg with the GIL released, so a producer thread hides it behind the GPU work.
Golden set wall-clock 1088 s -> 668 s, output byte-identical. The gain is bounded by
`max(decode, analysis)` per frame, so it is large on OCR-heavy videos (-37% on
`0450713odlisa`) and small on decode-bound ones (-11%).

**Unwrapping `nn.DataParallel`** around EasyOCR's detector and recognizer. On a
single device it adds no parallelism, but every forward walked the full parameter
and buffer list to validate device placement.

**Caching `load_setting_regions_config`**. Read-only config that was re-read from
disk for every single-digit OCR result (63,995 times in one profiled session).

## What did not work

**A change-detection gate on raw pixels.** The idea was to skip work when a
region's crop is unchanged since the previous frame. Measured over 3,000
consecutive frames, the fraction of frames where the raw crop is byte-identical to
the previous one:

| region | machine 1 | machine 3 |
|---|---:|---:|
| STAGE | **0.0%** | 0.9% |
| PEDAL | 35.8% | 2.2% |
| OCR regions | 0.0–73.7% | 1.0–42.7% |

H.264 decode leaves small pixel differences across the whole frame, including
opaque UI panels, so "unchanged" almost never happens exactly. Binarizing first
does not rescue it either (0.0% on machine 1 STAGE, 20.4% on machine 3 STAGE).

**Decode tuning.** Every configuration tried was slower than the OpenCV default
(4.88 ms/frame warm):

| configuration | ms/frame | note |
|---|---:|---|
| default | **4.88** | fastest |
| `CAP_PROP_N_THREADS=8` / `=16` | 9.10 / 9.02 | ~1.9x slower |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;8` | 5.56 | |
| `...=threads;8\|thread_type;frame` | 6.90 | |
| `...=hwaccel;cuda` | 6.66 | NVDEC output still has to come back to CPU and NV12->BGR |
| `...=hwaccel;d3d11va` | 6.62 | |
| `CAP_MSMF` backend | 59.61 | and frames differ from FFMPEG — unusable |

The pip `opencv-python` build has no CUDA modules at all (`cudacodec` is listed
under Unavailable, `cv2.cuda.getCudaEnabledDeviceCount()` is 0), so GPU decode
through OpenCV is not available without rebuilding it. Lowering the decode floor
means replacing the reader (PyNvVideoCodec, decord) and keeping crops on the GPU —
a separate project, not a configuration change.

**Skipping the CRAFT text detector, on machine 1.** EasyOCR's `readtext` runs text
detection before recognition, and detection is about half of each call. Calling
`Reader.recognize` with no bounding box (easyocr then treats the whole crop as one
text box) is 1.56x faster per call — 26.31 ms -> 16.85 ms — but on
`1131009何素珠od` it agrees with the current path on only 59.7% of 6,906 calls:

| outcome | count | share of calls |
|---|---:|---:|
| current path read a value, detector-free path read nothing | 2,452 | 35.5% |
| current path read nothing, detector-free path read a value | 304 | 4.4% |
| both read a value, values differ | 26 | 0.4% |

The current path's confidence on the 2,452 readings the other path loses has a
median of 1.000, so these are not marginal cases. The cause is visible in the
disagreements where both paths read something: `'15 60'` -> `'1560'`,
`'350 700+ 600'` -> `'35000500'`. Machine 1's value regions are 196x72 panel areas
holding both the large operating value and the small setting value, not single-line
crops — so detection is doing real work by locating one line inside the panel, and
treating the whole ROI as one box runs the elements together.

The flag therefore stays off by default (`"ocr_detector": "craft"`). It is per
machine because machine 3's regions are genuinely tight single-value crops
(`INCISION` 31x18, `REFLUX` 26x15), where the premise may hold; see
`tools/ab_ocr_detector.py` for how to re-run the comparison.

## Open findings not yet acted on

- On `1131009何素珠od` the current path returns nothing on 304 calls where the
  detector-free path reads a value with median confidence 1.000. Those readings are
  dropped from the output today, so detection is occasionally failing on text the
  recognizer can read.
- `region7` has no entry in `config/ocr_char_sets.json` and falls back to the
  default allowlist. It records 31 changes but makes 58 OCR calls on
  `2025-08-16_135759`, because the empty-result retry path fires almost every time.
- STAGE segment counts per video are inconsistent across machines: 11-16 on machine
  1, but 5-19 on machine 2 and 149-265 on machine 3 for comparable frame counts.
  Machine 2 looks under-detected and machine 3 over-segmented; both are threshold
  calibration questions, separate from performance.
- A tolerance gate (reuse the previous verdict when the binarized crop changed by
  less than a threshold) does reproduce the per-frame pattern sequence exactly at
  the right threshold, but the safe threshold differs by an order of magnitude per
  region: machine 1 STAGE tolerates 0.02 (42% of template scans avoided, 100%
  agreement) while machine 3 PEDAL only tolerates 0.001 (17.7% avoided; 0.01 drops
  to 93.7%). Template matching is currently only ~7% of runtime, so this matters as
  insurance for adding pattern regions rather than as a present-day speedup.
