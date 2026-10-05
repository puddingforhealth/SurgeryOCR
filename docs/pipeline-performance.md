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

**Skipping the CRAFT text detector.** EasyOCR's `readtext` runs text detection
before recognition, and detection is roughly half of each call. Calling
`Reader.recognize` with no bounding box — easyocr then treats the whole crop as a
single text box — is substantially faster, and substantially less accurate. Measured
with `tools/ab_ocr_detector.py`, which records the exact images a real run sends to
OCR and replays them through both paths:

| | machine 1 (1131009何素珠od) | machine 3 (0450713odlisa) |
|---|---:|---:|
| OCR calls compared | 6,906 | 3,234 |
| current path, ms/call | 26.31 | 45.19 |
| detector-free, ms/call | 16.85 | 16.43 |
| speed-up | 1.56x | 2.75x |
| **agreement** | **59.7%** | **51.5%** |
| current read a value, detector-free read nothing | 2,452 (35.5% of calls) | 782 (24.2%) |
| both read a value, values differ | 26 (0.4%) | 620 (19.2%) |
| current read nothing, detector-free read a value | 304 (4.4%) | 167 (5.2%) |

The current path's confidence on the readings the other path loses has a median of
1.000 on both machines, so these are not marginal cases. The cause is visible in the
disagreements where both paths read something:

```
machine 1:  A='15 60'        B='1560'       A='350 700+ 600'  B='35000500'
machine 3:  A='50 200'       B='50200'      A='30 37 30'      B='303730'
            A='90'           B='09'         A='28 40'         B='2840'
```

The configured value regions are **panel areas holding several text elements**, not
single-line crops — machine 1's `region1`-`region3` are 196x72 and carry both the
large operating value and the small setting value; machine 3's `ASP` is 167x34 and
`VAC` 167x42. Detection is therefore doing necessary work by locating one value
inside the panel, and treating the whole ROI as one box runs the elements together
or picks up a neighbouring glyph. The premise that the ROIs are already tight crops
is wrong for both machines.

The flag therefore stays off by default (`"ocr_detector": "craft"`). It is kept, per
machine, because the result above identifies a real opportunity rather than closing
one: **if a region declared a box per value instead of one box per panel, the
detector could be skipped for a 1.6-2.75x reduction in OCR cost**, which is 57-70%
of the runtime on videos with dense readings. That needs re-annotation with
`roi_picker.py` (one box per value) and a per-value box list in the region
declaration — not a code-only change, and an accuracy decision rather than a
performance one. `tools/ab_ocr_detector.py` re-runs the comparison after any such
change.

## Data-quality finding: machine 3 ASP / VAC regions span three values

Not a performance issue, but found while measuring one, and it affects output that
already exists.

The UNITY "Asp Flow" and "Vacuum" panels are ramp widgets showing three numbers —
start, current and end — above a triangle whose apex marks the current position.
The configured ROIs cover all three:

| region | box | size | values inside |
|---|---|---:|---|
| `ASP` | `[1748, 482, 1915, 516]` | 167x34 | `30` `37` `30` |
| `VAC` | `[1748, 568, 1915, 610]` | 167x42 | `350` `725` `600` |

On `0450713odlisa` this shows up directly in the output: 60 of 349 `ASP` readings
(17.2%) and 61 of 339 `VAC` readings (18.0%) are several numbers concatenated —
`'30 37 30'` 23 times, `'350 725 600'` 21 times. EasyOCR detects each number as its
own text box and the wrapper joins the results left to right, so the recorded
`ocr_text` is faithful to the ROI; the ROI is what is wrong.

The remaining single-number readings are the more serious problem, because nothing
records *which* of the three numbers was read. A sample of consecutive `ASP`
readings:

```
11293 '3'    11423 '5'    11591 '37 30'   11615 '30 37 30'   11724 '5'
11829 '12'   11915 '9'    11931 '10'      11954 '30 37 30'   11990 '33'
```

So the `ASP` and `VAC` series on machine 3 cannot currently be read as a time series
of one quantity. Fixing it means deciding which of the three numbers is wanted and
re-annotating a box around just that one — which is the same change that would allow
skipping the text detector, so it buys data quality and a 2.75x OCR speed-up at
once. Whether the centre value's box can be fixed at all depends on whether the ramp
apex moves horizontally; that needs checking against the machine.

Machine 1/2 regions have the same shape of problem far more rarely: `region2` and
`region3` show a space in 0.2-0.3% of readings, `region5` in 2 of 4.

## Open findings not yet acted on

- The current path returns nothing on 304 of 6,906 calls (machine 1) and 167 of
  3,234 (machine 3) where the detector-free path reads a value with median
  confidence 1.000. Those readings are dropped from the output today, so detection
  is occasionally failing on text the recognizer can read — worth revisiting with
  easyocr's `min_size` (default 20), which filters small detected boxes and may be
  rejecting text in the smallest regions (`REFLUX` is 26x15).
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
