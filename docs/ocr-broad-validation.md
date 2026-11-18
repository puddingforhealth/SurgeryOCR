# Machine 1/2: larger local GPU experiment

Evidence date: 2026-11-16. Local NVIDIA RTX 3070 8 GiB; no remote GPU workload
was started. Existing production models, defaults and saved analyses were not
replaced by this experiment.

This is the preserved experimental checkpoint. Subsequent deployed-adapter
regression and full-video integration are documented in
[integrated OCR profiles](ocr-production-integration.md); the original evidence
and annotation revisions below remain unchanged.

## Data and protocol

716 original RGB crops were visually reviewed by the assistant, with matching
current binary inputs, across 57 source videos. After excluding ambiguous,
multiple-target, duplicate and boundary examples, 597 were eligible: 154 train,
102 validation and 341 test. The test includes 304 newly collected examples
from 33 previously unused videos; the rest are seed regression examples.

The train/validation pack was frozen before fitting. No test image was loaded
by the trainer. Duplicate test inputs matching fit/validation or an earlier
test example were excluded by hash before test prediction. Original reviews
and exclusions remain in the full ledger. No input/label collisions were found.
Case selection deliberately includes low-confidence readings, negative values,
decimal values, and high-confidence controls; it is not random sampling of all
video frames or a population accuracy estimate.

Six trials crossed learning rates 0.00001, 0.00003, 0.0001 with seeds 1111 and
2222. Each ran 20 epochs, batch 16, on the same 154 training examples. Each
checkpoint was selected by validation. Validation then compared the existing
CRAFT call against direct-ROI recognition and an explicitly separate
conservative policy. No test predictions were generated before the final
method/weight selection was written to the selection ledger.

The selected weights used learning rate 0.00003 and seed 1111, with direct-ROI
recognition and conservative routing. Validation was 101/102 versus 90/102 for
the original CRAFT path, with no lost baseline-correct example. This is a
workflow comparison, not proof that a weight change alone produces the gain.

## Primary frozen-label test result

| Method | Exact / 341 | Baseline-correct examples lost |
|---|---:|---:|
| Original CRAFT / original weights | 268 | — |
| Selected fine-tuned weights / direct ROI, without routing | 329 | 4 |
| Same selected weights / direct ROI / conservative routing | 333 | 0 |

The final route improved 65 examples. All 268 baseline-correct examples remained
correct in this selected test. High-confidence controls had no regressions.
On the newly collected video cohort the primary result was 239/304 to 297/304.
Missing minus signs fell from 3 to 0 over 56 negative labels; missing decimal
points fell from 43 to 1 over 120 decimal labels. These symbol metrics do not
guarantee that every remaining digit is correct.

The paired accuracy improvement's video-cluster bootstrap 95% interval was
approximately +14.7 to +23.6 percentage points. It resamples source videos, not
individual adjacent crops. This interval does not cover annotation errors,
selection bias, unseen firmware/layouts, or all sources of model uncertainty.

The first 5-step local GPU smoke test used about 150 MiB peak PyTorch allocated
memory at batch 8. Larger batch-16 trials used about 215 MiB, under an explicitly
set 2 GiB PyTorch allocator cap. CUDA context/driver overhead and desktop memory
are additional. These measurements concern the small English gen2 recognizer,
not CRAFT training or VLM fine-tuning.

## Routing behavior and remaining failures

`utils/ocr_prediction_policy.py` never constructs a numerical value or inserts a
point/sign. For the verified machine-1/2 regions:

1. Preserve a valid original prediction with confidence at least 0.95.
2. Otherwise use a valid new prediction.
3. If the new result is invalid but the original is valid, retain the original.
4. If neither is valid, retain the original untrusted text and flag review.

The current layout's `region6` uses two decimal digits; other reviewed regions
use integer strings. Unknown regions and machine 3 are rejected by this policy.
This rule is a domain-specific experiment, not a format guarantee for future
firmware. Disagreement/invalid results flag review. A confident valid original
can still be wrong and be preserved; agreement is not ground truth either.

The raw-model result and its four regressions are retained separately. The
policy's preservation of confident originals is intentional behavior, not
evidence that the new model alone learned all easy cases. Some remaining
examples lose a dot or a digit detail during binarization, or acquire border
noise resembling punctuation. More epochs cannot recover information absent
from the model input. RGB/grayscale preservation and revised ROI preprocessing
are candidates for a future experiment, using new untouched test videos.

## Annotation errata: separate from the frozen score

All labels are assistant-reviewed, not independently human-verified. A second
visual review after evaluation found four original label mistakes (including
leading `0` misread as `1`, and `18` misread as `13`). The original full ledger,
frozen test pack, predictions and primary scores above are preserved unchanged.

The current working dataset records the corrections with original labels and
revision notes, and `label-errata.json` records crop hashes and review provenance.
A secondary view of the **same predictions** scores 271/341 original and
337/341 routed after those corrections. This is not another independent test
and must not replace the primary score without disclosing the revision. No
weights, thresholds, training or method selection were changed after the test.
Remaining disagreement still needs independent human review before deployment.

## Compare through the actual OCR call

```powershell
python tools/compare_ocr_models.py --data DATASET --weights CANDIDATE/numeric_m12.pth --output NEW_COMPARISON --split val --device cuda --desktop-gpu --candidate-detector whole_roi --prediction-policy conservative
```

| Flag | Meaning |
|---|---|
| `--data`, `--weights`, `--output` | Required reviewed dataset, candidate weights and new output folder |
| `--split val\|test` | Default validation; select on validation before test |
| `--device cpu\|cuda` | Default CPU |
| `--desktop-gpu` | Explicit Windows desktop exception; strict shared-GPU gate remains default |
| `--candidate-detector craft\|whole_roi` | Default CRAFT; original baseline always uses CRAFT |
| `--prediction-policy raw\|conservative` | Default raw; report weight/route changes separately |
| `--base-model-dir PATH` | Existing English gen2 and CRAFT weights; default EasyOCR model cache; downloads disabled |

The tool retains field allowlists, confidence threshold 0.5 and the original
empty-result trim retry. It does not simulate stage activation, cache hits,
debounce or the complete video pass. It uses two CPU threads and, for CUDA,
the first local GPU with a 2 GiB allocator cap. Use no CUDA visibility remapping
with this current single-GPU comparison tool.

`comparison.json` includes per-image predictions, confidences, candidate policy,
improved/regressed IDs, per-machine/region/selection/cohort metrics and the
video-cluster interval. For a conservative run it also retains the raw candidate
text, confidence, chosen source and review flag. The local experiment directory
contains the frozen fit pack, six runs, validation-only selection ledger,
frozen test pack, model hashes, label errata, full review ledger and comparison.
Images and weights are private local artifacts, not repository assets.

## Verification

The maintained full unittest suite passed 235 tests after the desktop GPU,
paired-comparison and routing additions. New Python files passed Ruff lint and
format checks. Dataset image hashes, fit/test boundaries, the frozen fit hash,
model hashes and report image references were checked. The isolated CPU ONNX
environment and previous benchmark remained separate from GPU training.
The repository-wide gate is not green: the earlier full check reported 207
existing lint findings outside these additions. No broad reformat or unrelated
test weakening was used to conceal those findings.

The reusable lesson is input-path alignment: improving a recognizer on a whole
ROI does not prove improvement on CRAFT-selected crops. Compare the actual
caller, show detector/format/routing changes separately, and retain paired
regressions and annotation revisions. Assistant labels and a selected error
corpus remain distinct from independently human-reviewed acceptance data.
