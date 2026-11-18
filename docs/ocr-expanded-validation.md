# Expanded fixed-model pipeline validation

This protocol extends the [2026-11-16 integration checkpoint](ocr-production-integration.md).
The fixed-model full-video comparison and expanded machine-1/2 scoring are
complete. This is a research checkpoint, not independent deployment acceptance.
Private videos, labels, reports and working scripts live under
`build/ocr-expanded-validation-20261116/`; they are excluded from Git.

## Scope

The controlled plan contains four confirmed machine-1 test-source videos, six
machine-2 test-source videos and the two available machine-3 development videos.
Machine-1 sources are disjoint from fitting/validation sources but were consumed
in earlier research. The six machine-2 sources were newly selected for this
comparison; one is a short recording. Machine-3 results remain development
diagnostics, not unseen-video accuracy. Known machine IDs are pinned for both
passes so automatic discovery does not change the input path between passes.

The installed bundle remains `2026.11.16.1`. Neither weights nor thresholds are
tuned from this test. Both full pipelines process the original decoded videos
in RAM mode, with separate output directories and no screenshot/frame-cache
export. The existing completed machine-3 and one machine-1 comparison are reused
with preserved provenance instead of rerunning identical work.

The first collection produced 4,136 field-frame crops: 2,000 from machine 1
and 2,136 from machine 2. Its nominal quotas were approximately 4,100; the short
recording added independent timeline controls when older event coverage was
sparse. It samples older low-confidence events,
negative readings, decimals, high-confidence controls and independent timeline
positions. A supplementary collection targets distinct binary inputs containing
older negative predictions and nonzero-decimal candidates from the long
machine-2 recordings. Selection categories are candidates, not ground truth.

Labels are read visually from original RGB crops, without displaying new
predictions. Wider same-frame context is retained when the original ROI clips
a character; the actual OCR input remains unchanged. Ambiguous/multiple-target
images are recorded as exclusions. Exact decoded-pixel copies of fitting or
validation images are excluded before scoring. Within-test duplicates and input
label collisions are reported separately; repeated pipeline states are not
silently presented as independent recognition trials. Labels are assistant
reviewed and have not received independent human confirmation.

## Decoder and resource verification

Interactive indexed access and the original crop collector used OpenCV/FFmpeg
with PTS verification and random seeks. Production analysis reads frames
sequentially through `utils.pipeline_utils.iter_video_frames`. The resumed batch
collector uses that same iterator with one decoder thread and saves only the
selected crops. A check across three videos passed 36 exact RGB comparisons
against the prior indexed crops; maximum timestamp difference was
`5.684341886080802e-14` seconds. This check does not claim to verify every source
frame. The first verification attempt had an invalid sampling assumption that
each source supplied 12 crops before frame 6,000; source selection was corrected
while retaining all 36 pixel/time assertions.

The owner selected balanced mode for this job: a shared limit of eight of the
desktop's 16 logical CPUs, below-normal priority, two PyTorch CPU threads, one
batch decoder thread and a 0.25 PyTorch GPU allocation fraction. The earlier
conservative two-CPU phase and competing desktop work invalidate wall-time
speed comparisons. Affinity/priority checks include future analysis children.
Process control verifies the private task directory/output path before applying
settings; it does not stop or lower another session's processes.

Private adjustment command (CPU affinity/priority apply live; GPU allocation
settings apply when a new configured worker starts):

```powershell
python build/ocr-expanded-validation-20261116/resources.py --mode balanced
python build/ocr-expanded-validation-20261116/resources.py --mode background
python build/ocr-expanded-validation-20261116/resources.py --mode full
```

Optional resource flags are `--cpu-threads N`, `--logical-cpus ID,ID,...` and
`--gpu-memory-fraction FRACTION`. The controller validates IDs and ownership.
The existing full worker already uses a 0.25 GPU fraction; changing that value
cannot retroactively resize a running CUDA allocator. Voice inference was
already stopped and was not restarted. An existing cloud H100 was inspected
read-only and was fully occupied; no cloud workload was added or stopped.

## Continuous machine-3 diagnostic checkpoint

Twelve deliberately selected 32-frame windows provide 384 visual labels. They
include known difficult backgrounds and reported value changes; the result is
not a population estimate or full-video event-recall measurement.

| Observation | Legacy | Enhanced |
|---|---:|---:|
| Exact field-frame values | 303/384 | 362/384 |
| Matched real changes at the exact frame | 0/22 | 22/22 |
| Unmatched output transitions | 2 | 26 |

The enhanced route improves 81 labels and regresses 22. Its remaining errors
are 14 rejected readings exposed as blanks and eight missing minus signs.
Unmatched transitions include abstention/recovery boundaries, not only wrong
numerical values. This demonstrates why isolated crop accuracy alone does not
establish stable temporal output. It does not justify a new filtering threshold
without additional independent sequence validation.

## Completion gates and GUI outputs

The final report requires byte-identical stage JSON and frame-clock JSON for
every pair, equal decoded frame counts, equality to the container's declared
source frame counts, and zero clock drops. Equal incomplete runs cannot satisfy
source completeness. All annotation IDs, image hashes and labels must be frozen
before primary scoring; original frozen results remain separate from any later
annotation errata.

Completed runs are incrementally published to the owner's D-drive comparison
library under `runs/ocr-comparison-20261116/{legacy,enhanced}`. Each video has its
matching analysis directory. NTFS hard links provide the existing videos without
allocating another video copy. Original videos and analyses remain in place.
`Open-Enhanced.ps1` and `Open-Legacy.ps1` launch the existing GUI against the
selected result root. `library.json` lists only completed published analyses.
The self-contained temporal diagnostic report is published under
`diagnostics/temporal-machine3/comparison.html`.

Reproducibility artifacts include `plan.private.json`, frozen protocol/model
metadata, `progress.json`, per-pass logs, decoder verification, resource-control
records and the final `complete-video-verification.json`. The latter is created
only after all complete-source and pair-invariant checks pass. Full-video
execution still does not provide human ground truth for every decoded frame.

## Expanded machine-1/2 results

The frozen ledger contains 4,702 assistant-reviewed crops: 4,313 eligible,
364 ambiguous exclusions and 25 fitting-image duplicates. Eligible examples
cover 4,204 distinct video frames and 3,481 distinct binary inputs grouped by
machine and field, with no conflicting labels for an identical input. They
include 426 negative values, 1,913 decimals (1,282 nonzero), 1,343 saved
confidence values below 0.8 and 1,730 at least 0.95. Categories overlap.
The frozen ledger SHA-256 is
`9c6c3af1fead263d6021624b42a0cd898e980e99fbf7183286e3f73c417032aa`.

Training remains **154 train / 102 validation**. No weights were refitted or
thresholds tuned for this expansion. The historical 341-example crop test and
the new complete-pipeline test answer different questions; their scores must
not be pooled or presented as a model-only training gain.

| Machine / scope | Samples | Legacy exact | Enhanced exact | Improvements | Regressions |
|---|---:|---:|---:|---:|---:|
| Machine 1, all eligible | 1,811 | 1,311 (72.39%) | 1,472 (81.28%) | 168 | 7 |
| Machine 2, all eligible | 2,502 | 1,953 (78.06%) | 2,113 (84.45%) | 164 | 4 |
| Machine 1, stage allows activation | 1,440 | 1,250 (86.81%) | 1,411 (97.99%) | 168 | 7 |
| Machine 2, stage allows activation | 2,130 | 1,880 (88.26%) | 2,040 (95.77%) | 164 | 4 |

There are 743 stage-disabled samples; both pipelines match 134 of those,
including blanks. Retain them in the unconditional metric and show the active
subset separately. Active-subset scoring still includes change triggers,
memoization, routing and blank behavior; it is not recognizer-only accuracy.
High-confidence controls contain two regressions, so the earlier small-test
claim of no lost baseline-correct cases does not generalize to this set.

| Challenge subset | Samples | Legacy exact | Enhanced exact | Regressions |
|---|---:|---:|---:|---:|
| Negative values | 426 | 400 | 410 | 0 |
| Nonzero decimals | 1,282 | 1,063 | 1,215 | 6 |
| Saved confidence below 0.8 | 1,343 | 953 | 1,106 | 7 |
| Saved confidence at least 0.95 | 1,730 | 1,263 | 1,354 | 2 |

All 12 paired sources passed completeness: 499,988 source frames and 999,976
processed frames across 24 passes, with equal stage/clock JSON and zero recorded
clock drops. Wall times remain unsuitable for a controlled speed comparison.

Private self-contained reports under `reports/` separate `machine12.html` from
`machine3.html`. The machine-1/2 page includes every eligible crop, same-frame
wider context when available, per-video and challenge tables, and filters for
improvements, regressions and unresolved errors. `excluded.html` retains the
excluded evidence. `summary.json` stores the exact counts and provenance.
Reports are published beside the GUI library under its `reports/` directory.
The old 16/8-case integration report is retained as historical evidence, not
the main machine-1/2 result.

Machine-3 point scores retain their prior frozen labels. One current working
label differs from that historical ledger; the difference is recorded separately
rather than silently replacing the old score. Its two development videos and
384-frame temporal diagnostic remain separate from the new machine-1/2 test.

Regression verification reported **296 tests OK in 58.416 seconds**. No
production pipeline, model weights, trigger thresholds or test expectations
were changed to produce these reports. Native GUI visual inspection was not
completed because the computer-use Node runtime failed to start; result-path
and machine-resolution checks and GUI preload logs are separate evidence.
