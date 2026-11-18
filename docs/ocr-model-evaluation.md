# Machine 3: low-cost recognizer evaluation

Evidence date: 2026-11-16. These are local CPU experiments on private crops; no
images were sent to a hosted OCR service and no remote GPU workload was started.

## Measured result

The diagnostic set has 91 eligible crops from two machine-3 source videos:
89 numerical strings and two blanks. Sixteen multiple-target/ambiguous crops
from the original 107-image review are excluded. Labels were visually assigned
by the assistant and are not independently human verified. Repeated values and
selected errors make this unsuitable as a population accuracy estimate.

| Recognizer/method | Full alphabet exact | Numerical decoding exact | Warm CPU inference median |
|---|---:|---:|---:|
| EasyOCR 1.7.2 English gen2, whole ROI | 34/91 | 41/91 | 30.6 ms |
| PP-OCRv5 English mobile ONNX | 79/91 | 88/91 | 21.2 ms |
| PP-OCRv6 tiny ONNX | 85/91 | 85/91 | 3.3 ms |

Both Paddle candidates read all 26 CDE, 15 INCISION, and 12 INFLOW examples
correctly in this set. The v5 numerical decoder missed two visible PEL minus
signs and read one blank as `1`. v6 retains fewer PEL signs, despite its lower
latency. These are recognizer/decoder comparisons; none includes production
debounce, template caches, stage activation, or value-format correction.

Full-alphabet output and numerical decoding are reported separately. The latter
allows only `+-0123456789.` while keeping CTC blank; it does not infer decimal
placement or fill missing signs. It can change mistakes, so the full-alphabet
result must not be hidden. EasyOCR uses its documented allowlist; the ONNX
experiment masks nonnumerical output scores before greedy CTC decoding. There
is no output-confidence threshold in these new comparisons.

Preprocessing is intentionally model-native: EasyOCR receives grayscale input
with minimum height 48 and then its own resizing/contrast behavior; Paddle
receives BGR color, height 48, normalized to [-1,1], zero-padded to at least
width 320. Neither receives a binary image. These differences are part of the
candidate methods, not a controlled ablation of model architecture alone.
CPU timing used two threads, excludes model loading/preprocessing, and is
machine-specific. Re-measure in the deployed pipeline before capacity planning.

## Provenance and cost

PP-OCR is a specialized OCR family, distinct from PaddleOCR-VL. EasyOCR's
English gen2 recognizer uses convolutional features, bidirectional LSTMs and
CTC. PP-OCRv5's mobile family uses lightweight visual features with a more
modern text-sequence recognition/training recipe; its
[mobile training configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml)
combines CTC and an auxiliary sequence head during training, with CTC decoding
for the exported recognizer. A detector is separate in both OCR families;
these ROI comparisons did not need a full-page detector.

Local weight inspection found about 3.78 million state-tensor elements in
English gen2, 1.94 million ONNX initializer elements in v5 English mobile, and
1.10 million in the converted v6 tiny artifact. These are inference-artifact
element counts, not exact comparable trainable parameter counts: conversion
can fold batch normalization and retain constants or remove training heads.
They nevertheless show the scale: a few million elements and a few MB.

By comparison, [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
is a 0.9-billion-parameter vision-language model for document parsing, text,
tables, formulas and related tasks. FP16 weights alone at that scale are about
1.8 GB; inference state, image tokens, activations and fine-tuning optimizer
state are additional. It combines visual encoding with language generation,
rather than specializing solely in transcribing a small fixed numerical ROI.
A larger parameter count is not evidence of better minus/decimal fidelity on
this layout. The local measured accuracy, latency and rejection behavior are
the appropriate comparison. An 8 GiB local GPU readily handled the small
English gen2 fine-tuning experiment; that does not imply an equivalent full
fine-tune of a VLM will fit.

| Artifact | Measured file size | Verified SHA-256 |
|---|---:|---|
| `en_PP-OCRv5_rec_mobile.onnx` | 7,872,351 bytes | `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8` |
| `PP-OCRv6_rec_tiny.onnx` | 4,489,813 bytes | `e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b` |

Models came from the versioned `v3.10.0` RapidAI/RapidOCR ModelScope distribution,
with hashes checked against the upstream [model manifest](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml).
RapidOCR's [model licensing notes](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)
identify the upstream model and conversion attribution. Preserve that attribution
if redistributing weights. This change does not distribute those weights.

The [current PaddleOCR recognition documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md)
lists v6 tiny/small/medium and the v5 English mobile model. Its v6 and v5 benchmark
percentages use different evaluation sets, so those percentages cannot rank the
candidates for our surgical overlays. The local comparison is more relevant,
but remains small and requires fresh-video verification.

Recommendation: test v6 tiny as a cheap primary candidate and v5 English mobile
as an accuracy-oriented candidate or fallback. Triggering a fallback only on low
confidence is insufficient: also consider model disagreement, invalid numerical
format, and point/sign evidence. Calibrate any reject rule on validation data.
Do not deploy a cascade or accept two agreeing models as ground truth without
testing correlated errors. No cascade is implemented yet.

[PARSeq-tiny](https://github.com/baudm/parseq) is another scene-text candidate if
these lightweight recognizers still fail on new backgrounds; it has not been
benchmarked here. [TrOCR-small-printed](https://huggingface.co/microsoft/trocr-small-printed)
has 61.4 million parameters and targets text lines; it is a higher-cost secondary
experiment rather than the first choice for a few numeric glyphs. Generic VLMs
are more appropriate for assisted annotation than default per-frame inference
until their numerical fidelity and latency have been measured.

An additional path is training on real numerical foregrounds over varied local
surgical backgrounds, with opacity, blur, compression, and color variation.
Keep real held-out videos: synthetic compositing can fail to reproduce the
actual overlay rendering and cannot establish real-world accuracy by itself.

## Reproduce the ONNX comparison

```powershell
python scripts/setup_ocr_benchmark.py
.venv-ocr-benchmark/Scripts/python tools/benchmark_ocr_onnx.py --data REVIEW_FOLDER REVIEW_EXPANSION --model LOCAL_MODEL.onnx --output NEW_RESULT_DIR
```

On Linux/macOS use `.venv-ocr-benchmark/bin/python`. The setup uses `uv` and an
isolated environment, with versions in `config/ocr_benchmark_requirements.in`.
No PaddlePaddle GPU installation is needed. Download the chosen model from the
upstream manifest and verify its listed SHA-256 before passing it to the tool.

The benchmark takes `--data` (one or more reviewed folders), `--model` (local
CTC ONNX with an embedded RapidOCR character dictionary), `--output` (a new
directory), and `--threads` (default 2). It accepts `labels.jsonl` records with
`id`, `region`, `image`, `image_sha256`, `label`, and `status`; only
`reviewed_numeric` and `reviewed_blank` are evaluated. It verifies image hashes,
rejects duplicate IDs, uses CPU execution only, and saves per-image predictions,
per-region metrics, model hash, runtime version and timing in `evaluation.json`.
The embedded dictionary length must match the model head; incompatible
architectures fail rather than using a guessed dictionary.
