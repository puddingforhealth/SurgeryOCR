# Machine-3 decoded-pixel color study

Evidence date: 2026-11-17. This is an exploratory diagnostic on the existing
two development videos, not a new OCR accuracy comparison or a production
preprocessing change. Private artifacts are under
`build/ocr-m3-color-study-20261117/`.

## Sampling and provenance

The study reads the 566 existing RGB crops and checks each PNG against its
label-ledger image SHA-256. Sixteen numeric crops were selected: for each video
and each of eight fields, choose the numeric crop with the highest 80th
percentile RGB mean on the four-pixel border. This favors bright surrounding
backgrounds; it is not random sampling.

Assistant visual annotations locate 68 stroke-interior points and 64 background
points. Initial approximate coordinates were reviewed on enlarged overlays;
misplaced points were corrected, with the draft retained. Consequently, this
is exploratory annotation, not blinded or held-out threshold validation.
There are no complete glyph masks or independently human-verified pixel labels.

## Observations

Values below are 8-bit decoded RGB. They are not the original instrument
framebuffer colors: video compression, antialiasing and background blending
can change them.

| Sample | RGB median | RGB 10th percentile | RGB 90th percentile |
|---|---|---|---|
| Stroke points | (255, 251, 247) | (240.4, 236.7, 230.0) | (255, 255, 255) |
| Background points | (106.5, 89.5, 83) | (43.9, 35, 28) | (180.9, 167.1, 174) |

Stroke points are predominantly bright and nearly achromatic, with a slightly
warm median near `#FFFBF7`. Their median OpenCV HSV saturation is 6/255; sampled
background saturation is higher. Hue alone is poorly informative for these
nearly white pixels. Per-crop distances between the mean text/background Lab
vectors range from 35.41, with median 54.825 (Euclidean CIE76 units). These
descriptive distances are not a statistical-significance test: fields and
frames share only two source videos.

Color separation is useful in many inspected numeric crops, but background
reflections and graphics overlap the text's color distribution. The 161
existing blank-labeled crops contain 110 pure-white `(255,255,255)` pixels in
total. The most affected crop contains 49. Blank-crop diagnostics do not measure
the entire nontext area inside numeric crops.

## Candidate masks and limits

HSV uses OpenCV uint8 units: S and V are 0–255. These conditions were explored,
not selected as a production setting.

| Mask | Stroke points retained | Background points retained | Blank crops with at least 9 selected pixels |
|---|---:|---:|---:|
| V >= 160, S <= 70 | 68/68 | 10/64 | 113/161 |
| V >= 180, S <= 60 | 68/68 | 7/64 | 100/161 |
| V >= 200, S <= 50 | 68/68 | 3/64 | 78/161 |
| V >= 220, S <= 40 | 66/68 | 0/64 | 60/161 |
| V >= 230, S <= 30 | 65/68 | 0/64 | 30/161 |

The V >= 200/S <= 50 mask selects 1.2104% of all pixels in blank crops. A small
pixel fraction can still form distracting strokes. The nine-pixel diagnostic
cutoff is arbitrary and is not an OCR false-positive decision. Tightening the
threshold reduces interference but already removes some annotated stroke points.
Point retention does not measure complete-stroke recall, decimal/minus survival,
temporal stability or recognized numerical accuracy.

In inspected case 11 (`m3-2-ASP-14502`), the original reads `30`. The candidate
mask visually joins a reflection to the left of the `3`, making it resemble
an `8`. This is an observed mask-shape ambiguity, not a measured OCR prediction.

The self-contained `report.html` shows original crops, sampling coordinates,
candidate masks and blank-background interference. `measurements.json` retains
all point RGB values, the full-crop mask counts and the input-ledger hash.
No source videos, labels, models, production configuration or prior results
were modified.

## Next experiment

A color-derived soft feature is a plausible supplement to the RGB input. Compare
RGB, a color mask and their combination on a separately frozen sequence set,
including blank backgrounds, negative signs, decimals and bright reflections.
Training/validation/test sources must remain separate. No such OCR experiment
has been performed here, and color separation alone does not establish an
accuracy improvement.

Verification covered all 566 input hashes, visual inspection of the 16 numeric
overlays and eight high-interference blank examples, five candidate masks,
monotonicity checks, synthetic color-mask checks and output/publication hashes. These checks
do not constitute independent segmentation ground truth.
