# Single-recognizer trial for machines 1 and 2

The trial bundle, version `2026.11.18.single.2`, uses one fine-tuned English
gen2 recognizer on numeric regions 1 through 6. BSS (region 7) continues to use
the original reader alone. Some real BSS panels display the word `Full`, which
was excluded from the numerical labeling exercise. Blank reads retain the
existing same-model border retry. This bundle is now the backend default. Machine 3 retains its existing weights.

## Collection and selection

We visually reviewed RGB crops with wider right-edge context while the model
received the original binary ROI. Two known training sources and 18 additional
machine-2 sources supplied new development and test examples. The 18 additional
video files had distinct sizes with no size matches among prior source videos.
Multi-value panels and uncertain or partial strings were excluded. The labels are
assistant-reviewed with no independent human verification.

The initial collection comprised 428 crops and a further 79 targeted crops.
For TORS, querying historical OCR events had yielded mostly zero or blank.
Sampling pedal intervals yielded another 252 crops. Selecting two-glyph candidates
during pedal level 3 yielded 137 more crops, including visually confirmed 33 and
38. Use the raw video for this sampling; old change records alone miss valid
transient numbers. Deduplication used binary pixels and excluded copies of the
earlier frozen challenge inputs. Augmented examples were restricted to training.

Three training experiments started from the previously selected EasyOCR weights.
The first used 419 training rows (265 newly reviewed) from 27 videos and 144
validation rows. The second applied one-pixel augmentation. The third included
the additional pedal samples and had 581 training rows and 166 validation rows.
On the expanded validation set the third candidate scored 163/166, the first
162/166 and the augmented candidate 159/166. The third candidate scored only
4,141/4,313 on the reused challenge set. The first candidate scored 4,214/4,313.
The first candidate was subsequently fixed for prospective testing. Its challenge
set score is selection evidence, not independent acceptance.

## Results for the fixed trial weights

| Scope | Original | Previous dual | New single |
| --- | ---: | ---: | ---: |
| Reused challenge, all 4,313 | 3,796 | 4,186 | 4,214 |
| CDE, 1,915 | 1,478 | 1,844 | 1,857 |
| TORS, 327 | 325 | 315 | 317 |
| Blank, 236 | 236 | 218 | 221 |
| Negative, 426 | 400 | 410 | 423 |
| High saved confidence, 1,730 | 1,551 | 1,714 | 1,716 |
| Independent confirmation, 193 | 163 | 186 | 185 |

The prospective confirmation sampled eight additional machine-2 videos after
fixing the trial weight hash. Of 219 crops, 21 were ambiguous and five exactly
matched development pixels. Against the dual model the single model gained one
correct read and lost two. Several residual errors involved `0.00` and blank
panels. These targeted, correlated observations do not establish population
accuracy or statistical equivalence. The new confirmation contains no machine-1
videos and no BSS word-state evaluation.

Two losslessly encoded 720-frame excerpts ran through the normal RAM pipeline
with an explicit machine ID. On machine 1 both pipelines matched 17/29 labeled
points. On machine 2 the trial matched 71/74 and the dual route 72/74. Within
each excerpt the stage and frame-clock files matched byte for byte. A short clip
can start in the middle of a state; these checks exercise integration and are
not full-video accuracy benchmarks.

## Running the trial

Pass the extracted `single-bundle` directory to the updated checkout:

```powershell
python tools/install_ocr_bundle.py --doctor --destination single-bundle
python run_pipeline.py --video VIDEO.mp4 --mode ram --output-dir runs/single-trial --ocr-profile enhanced --ocr-bundle single-bundle
```

The startup log and `run_stats.json` should report `easyocr-single-numeric` for
machines 1 and 2. Model SHA-256 is
`0fda44ca8399aadd437a37078f333b4f8c592c18085aa70362aed11e76a6abbc`.
The local trial archive and per-sample evidence are in
`build/ocr-single-20261118/`. Use `confirm/results.json` for the prospective
comparison and `confirmed-video-check.json` for timeline checks.
