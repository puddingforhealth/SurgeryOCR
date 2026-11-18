# Reviewed numerical OCR datasets and fine-tuning

The OCR training tools operate on separate, private crop datasets. They do not
change the production OCR pipeline, configuration, model selection, or analysis
JSON files. No command submits a cloud job implicitly.

## Dataset contract (version 1)

```text
dataset/
  dataset.json
  records.jsonl
  images/rgb/<id>.png
  images/binary/<id>.png
```

`dataset.json` contains `{"schema_version": 1, "manifest": "records.jsonl"}`.
One JSON object per line in the manifest:

```json
{
  "id": "m1-v1-region2-f100",
  "video_id": "m1-v1",
  "machine_id": 1,
  "split": "train",
  "frame": 100,
  "seconds": 3.33,
  "region": "region2",
  "roi": [1640, 445, 1836, 517],
  "images": {"rgb": "images/rgb/example.png", "binary": "images/binary/example.png"},
  "sha256": {"rgb": "<file SHA-256>", "binary": "<file SHA-256>"},
  "label": "-12",
  "status": "reviewed_numeric",
  "annotation_source": "assistant_visual_review_of_rgb",
  "human_verified": false
}
```

Labels are exact strings: `-12`, `0.03`, and `2.20` retain signs and decimal
precision. `reviewed_blank` requires an empty label. Only these two statuses are
eligible; ambiguous, multiple-target, unreviewed, and duplicate examples remain
in the review manifest but are excluded from training/evaluation. Source paths
and patient filenames belong in a separate private provenance map, not in the
portable manifest. RGB images preserve evidence; binary images reproduce the
existing machine 1/2 input using the recorded preprocessing configuration.

Split by source video before model development. Never distribute adjacent frames
or RGB/binary variants from one video across train/validation/test. Validation
selects the checkpoint; test is evaluated afterward. The validator rejects source
videos and identical eligible image hashes spanning splits. Held-out examples
take priority when excluding duplicates. A small, deliberately selected error
set is a diagnostic benchmark, not a clinical or population accuracy estimate.

Assistant image review is not independent human verification. The explicit
`--accept-assistant-labels` flag records that distinction. It does not relabel
them as human-verified or establish production readiness.

## Validate, fine-tune, evaluate

Use the main project environment installed by `scripts/setup.ps1` or
`scripts/setup.sh`. The implementation was verified with EasyOCR 1.7.2 and
PyTorch 2.8.0. It reuses the installed English gen2 architecture and alphabet;
provide the matching local `english_g2.pth` (normally in `~/.EasyOCR/model`).
There are no model downloads in the training command.

```powershell
python tools/ocr_finetune.py validate --data DATASET --accept-assistant-labels
python tools/ocr_finetune.py train --data DATASET --weights WEIGHTS/english_g2.pth --output OUTPUTS/smoke --device cpu --epochs 1 --max-steps 2 --accept-assistant-labels
python tools/ocr_finetune.py train --data DATASET --weights WEIGHTS/english_g2.pth --output OUTPUTS/train --device cuda --batch-size 8 --max-vram-gib 4 --accept-assistant-labels
python tools/ocr_finetune.py evaluate --data DATASET --weights WEIGHTS/english_g2.pth --output OUTPUTS/baseline-test --split test --accept-assistant-labels
python tools/ocr_finetune.py evaluate --data DATASET --weights OUTPUTS/train/best/numeric_m12.pth --output OUTPUTS/candidate-test --split test --accept-assistant-labels
```

Replace uppercase paths with actual paths. Every output directory must be new.
Successful validation prints split counts, source video IDs, sign/decimal counts,
and human-verification counts. Training writes finite loss and validation metrics
to `history.jsonl`, provenance to `run.json`, and checkpoints to `best/` and
`last/`. `interrupted/` records a GPU conflict, runtime failure, or interruption;
it is not a successfully completed run. No automatic resume is implemented.

`best/` contains an EasyOCR-compatible `.pth`, `.yaml`, and `.py` triplet. The
weight keys retain the `module.` prefix required by EasyOCR 1.7.2. This export
is tested by loading it through an unmodified CPU `easyocr.Reader`.

The training/evaluation path is whole-ROI recognition at height 64, with
aspect-preserving bicubic resize and edge padding, matching EasyOCR's recognizer
normalization. It preserves the complete pretrained output alphabet. Evaluation
masks nonnumerical logits, uses greedy CTC, and applies no confidence threshold
or automatic sign/decimal correction. This is a recognizer experiment, not an
end-to-end pipeline rerun. Compare untouched and candidate weights through the
same evaluator; historical pipeline output is not the controlled baseline.

### CLI reference

`python tools/ocr_finetune.py -h` and `<subcommand> -h` list all options.

| Command/options | Meaning/default |
|---|---|
| `validate`, `train`, `evaluate` | Dataset check, fine-tuning, held-out inference |
| All: `--data PATH` | Required version-1 dataset root |
| All: `--input binary\|rgb` | Default `binary`; `rgb` is converted to grayscale without thresholding |
| All: `--accept-assistant-labels` | Accept examples without independent human verification |
| Train/evaluate: `--weights PATH`, `--output PATH` | Required local weights and new result directory |
| Train/evaluate: `--device cpu\|cuda` | Default CPU; CUDA never silently falls back |
| Train/evaluate: `--desktop-gpu` | Explicit local Windows WDDM mode; checks memory headroom while allowing GUI processes; never enabled by prepared shared-Linux jobs |
| Train/evaluate: `--gpu-index N` | Physical GPU index, default 0 |
| Train/evaluate: `--batch-size N`, `--width N`, `--threads N` | Defaults 8, 256, 2; excessive aspect ratio is rejected rather than squashed |
| Train: `--epochs N`, `--max-steps N` | Defaults 20, 0 (no step limit); a short smoke run verifies execution only |
| Train: `--lr FLOAT`, `--seed N` | Defaults 0.0001, 1111 |
| Train: `--max-vram-gib FLOAT` | Default 4; PyTorch allocator cap, not total GPU isolation |
| Evaluate: `--split val\|test` | Default test; never evaluate training data under a held-out label |

## Prepare an rcs job without submitting it

```powershell
python tools/prepare_ocr_job.py --data DATASET --weights WEIGHTS/english_g2.pth --output JOB_FOLDER --accept-assistant-labels
rcs job pack --file JOB_FOLDER/rcs-job.json
```

This creates an explicit bundle containing only trainer code, eligible crops,
labels, and weights. It omits original videos, source filename maps, unrelated
repository files, and previous outputs. Job preparation accepts `--epochs`
(20), `--batch-size` (8), `--max-vram-gib` (4), and `--remote-python` (`python3`).
The remote interpreter must already have the main project dependencies; no
shared environment is changed. `--data`, `--weights`, and `--output` are required;
`--accept-assistant-labels` has the same meaning as in the trainer.

When the target is idle:

```powershell
rcs --target zettabyte job run --file JOB_FOLDER/rcs-job.json
rcs job logs --file JOB_FOLDER/rcs-job.json --follow
rcs job fetch --file JOB_FOLDER/rcs-job.json
```

The GPU gate requires three idle checks, no foreign compute processes,
utilization below 10%, and allocator budget plus 1 GiB of free memory. During
training it checks foreign processes every ten steps and stops its own work if
one appears. It never stops other processes or requests/releases GPU resources.
The job fails when busy; it does not automatically wait or resubmit. A startup
race is still possible on a shared GPU; only a dedicated allocation or hardware
partition provides isolation. CPU evaluations follow training and do not hold
the GPU. Separate outputs retain the untouched and candidate test results.

For an owner-authorized local Windows desktop experiment, add `--desktop-gpu`.
The observed WDDM driver lists ordinary GUI programs in its compute-process
query, so the strict Linux process gate would refuse indefinitely. Desktop mode
requires Windows, checks allocator budget plus 1 GiB free before starting and
at least 512 MiB headroom every ten steps, and retains the PyTorch allocator cap.
It does not promise process isolation or an unaffected desktop frame rate. It
does not stop applications. The shared-Linux default is unchanged and rejects
this Windows-only exception. Close competing compute work before opting in.

## Acceptance and cheaper alternatives

A local CPU pilot on the 2026-11-16 seed pack completed ten epochs (50 optimizer
steps). The 36 training, 39 validation and 38 test examples came from six
source-disjoint videos. Validation selected the checkpoint before the held-out
test was evaluated. Under the same whole-ROI evaluator, unchanged English gen2
weights scored 19/38 exact on test and the selected checkpoint scored 37/38;
all six negative test values were correct, while one of 16 decimal values lost
its point. Historical full-pipeline output scored 30/38 on those same labels,
but is not a controlled inference-path comparison. This establishes a useful
small-data experiment, not production readiness: labels are assistant-reviewed,
samples deliberately target difficult cases, and test has only one video per
machine. This initial pilot used CPU. Subsequent local RTX 3070 experiments are
documented in [the larger validation record](ocr-broad-validation.md); GPU
training and allocator behavior have not been measured on the shared H100.

Report exact numerical accuracy, character error rate, missing minus signs,
missing decimal points, and per-machine/per-region results. Do not promote a
checkpoint based on training loss alone. Keep a broader normal-case set to detect
regression and a fresh video set for final acceptance after model selection.

For machines 1/2, first inspect cropping margins and binary inputs for destroyed
punctuation. A field-specific point/sign presence detector, fixed-font glyph
matching with an explicit reject threshold, or recognition-only fine-tuning may
be cheaper than a larger model. They need the same held-out tests; automatic
decimal insertion must not conceal actual failures or invent plausible values.
Temporal agreement is useful only inside a stable-value interval, and must not
erase genuine rapid changes. These alternatives are recommendations, not new
production pipeline behavior implemented by this change.

See [the measured machine-3 comparison](ocr-model-evaluation.md) for low-cost
color scene-text alternatives and the boundaries of the current evidence.

## Verification record and reusable boundaries

Status: validated locally for the dataset/training/export/packaging paths;
not a production-model acceptance or a green whole-repository quality gate.
On 2026-11-16, all 226 maintained `check_*.py` unittest cases passed, including
18 OCR training/packaging tests. New Python files passed Ruff lint and format
checks. The repository-wide check script still reported 207 lint findings in
existing files; its broader pytest discovery was interrupted rather than
misreported as passing. No golden pipeline rerun was needed for this isolated
tooling change. At that checkpoint no GPU training had been performed; the
subsequent local GPU work is recorded separately.

```powershell
python -X utf8 -m unittest discover -s tests -p 'check_*.py' -q
```

Keep these distinctions in later iterations: source-video separation is the
independence boundary; confidence-based selection does not establish ground
truth; assistant annotation does not establish human verification; historical
postprocessed output does not replace a same-path weight baseline; and a local
allocator budget does not isolate a shared GPU. The tests reject split leakage,
changed input hashes, unreviewed/scalar-invalid labels, unacknowledged assistant
labels and busy GPUs, and load exported weights through unmodified EasyOCR.
Future model promotion still requires broader normal cases and fresh videos.
