# Agent handoff: current repository state

This is a short orientation for any agent joining the project. Read [`AGENTS.md`](../AGENTS.md) for durable rules and [`README.md`](../README.md) for the audit report and experiment values.

## State at handoff (2026-10-28)

### Latest release-branch validation

- Branch `audit/research-validation-release` contains the audited implementation, docs, tests, and compact JSON evidence artifacts; it was pushed to the project GitHub repository. Datasets, videos, checkpoints, environments, and build output were excluded.
- `npm run build` passed with Vite 8.0.2 in 778 ms. `npm run check:routes` rendered `/`, `/demo`, `/research`, and `/contact`.
- In the isolated `.venv-video`, with a project-local Ultralytics config directory, `python -m pytest tests -v --ignore=tests/test_api.py` passed 52 tests in 8.30 s. The full suite could not collect because FastAPI is missing from this video-only environment; no extra dependency was installed.
- The current notebook files were inspected and sanitized; their Kaggle config is built from runtime environment variables. A hard-coded Kaggle credential was found in the earlier notebook versions in Git history. Revoke/rotate that key: renaming and sanitizing the files does not remove the historical value. Do not rewrite shared history without explicit owner coordination.
- The real-video tracker processed 148 decoded frames; confidence 0.25 yielded zero detections and IDs. Confidence 0.001 yielded 450 raw boxes, all below ByteTrack's 0.1 low threshold. Both settings missed all six annotated crossings.
- Organization changes: Colab notebooks now live under `notebooks/training/` with descriptive names and safe Kaggle credential setup; root guides moved under `docs/`; the unlabeled `test_sample/` folder is now `qualitative_samples/`; Vite's tracked dependency cache was removed and ignored. Historical result JSON was preserved unchanged.

- Existing implementation preserved: FastAPI, React/Vite, YOLOv8 fish checkpoint, preprocessing methods, an RT-DETR timing benchmark, cross-domain helper, and video tracking code.
- Current verification: `.venv` Python 3.11.9; `python -m pytest tests -v` passed **57 tests in 11.30 s**. Warnings: Starlette/httpx deprecation, pytest class-fixture deprecation, and PyTorch sm_120. Tests validate code paths/schemas, not fish accuracy.
- The current Codex audit shell could not reproduce that result: configured `.venv\Scripts\python.exe` targets a missing Python 3.11 executable. System `python -m pytest tests -v` (Python 3.14.7) collected 14 items and stopped with 2 collection errors because `cv2` is missing; no tests ran. No dependencies were installed.
- Frontend: `npm run build` passed with Vite 8.0.2 in 715 ms; `npm run check:routes` rendered `/`, `/demo`, `/research`, and `/contact`. The route check is an SSR smoke test, not browser E2E.
- GPU: RTX 5050 Laptop, 8,151 MiB, compute capability 12.0, driver 610.74. Installed `torch 2.14.1+cu126` has no sm_120 kernels; a minimal GPU tensor op fails with `cudaErrorNoKernelImageForDevice`; `get_optimal_device()` falls back to CPU. No GPU training, inference, or reinstall was performed.
- The original source-export folder had no `.git`; this release branch was pushed from a separate clone at `.github-push-checkout`.

## Local evaluation

- `models/best.pt`: 5.96 MB, SHA-256 `af229161766094f3f002ff301b68001f6bec84492fe4f43c41c9f0838d1a5d23`; 13 fish classes.
- Dataset: 8,229 images (6,834 train / 698 valid / 697 test), 640x640. After polygon-to-box conversion: 14,144 / 1,193 / 1,129 instances.
- `valid_full`: 698 images / 1,193 instances; P 0.880, R 0.875, mAP@50 0.910, mAP@50-95 0.658. Not independent; AngelFish has 0 instances.
- `test_full`: 697 / 1,129; P 0.816, R 0.804, mAP@50 0.840, mAP@50-95 0.622. Not fully independent.
- `test_clean`: 399 / 639; P 0.809, R 0.803, mAP@50 0.841, mAP@50-95 0.636. Leakage-filtered, not source-independent.
- Leakage counts: 34 test images exact-match train (33 distinct hashes); 212 share a train source stem (202 distinct prefixes); 70 share a valid source stem (69 distinct prefixes); test_clean excludes 298 images.
- Historical notebook log remains P 0.865, R 0.874, mAP@50 0.908, mAP@50-95 0.666 (12/13 classes). Keep separate from new evaluations.
- `qualitative_samples` has no labels; 39/40 images are exact duplicates of Dataset images. It is not independent cross-domain evidence.

## Bootstrap uncertainty (completed)

- `src/evaluation/uncertainty.py`, `tests/test_uncertainty.py` (14 tests), `scratch/run_bootstrap_uncertainty.py`.
- The script captures exact Ultralytics `DetectionValidator` TP rows with `multi_label=True`; it does not re-match `model.predict` output. Stats cache: `outputs/uncertainty_stats/`.
- Point estimates reproduce saved metrics within `max|delta| < 4.6e-7`.
- 2,000 resamples, seed `20261027`, image-level 95% percentile intervals plus Roboflow source-stem cluster sensitivity.
- `test_full` mAP@50 0.840 [0.826, 0.874], mAP@50-95 0.622 [0.604, 0.670]; finite 1,908.
- `test_clean` mAP@50 0.841 [0.823, 0.880], mAP@50-95 0.636 [0.615, 0.694]; finite 1,644; cluster [0.824, 0.880] / [0.617, 0.693].
- `valid_full` mAP@50 0.910 [0.886, 0.939], mAP@50-95 0.658 [0.637, 0.698]; finite 1,997.
- Overall intervals require all fixed classes present in the resample. They do not fix leakage or prove source independence.
- Artifact: `outputs/bootstrap_uncertainty.json`.

## Enhancement uncertainty (completed)

- Same 399 `test_clean` images, raw-original, JPEG-95 control, CLAHE, gray-world, fusion; CPU.
- Absolute mAP@50 / mAP@50-95: raw original 0.841/0.636; JPEG-95 0.841/0.636; CLAHE 0.846/0.634; gray-world 0.812/0.613; fusion 0.815/0.611.
- Paired deltas vs JPEG-95 control: CLAHE +0.0052 [-0.0028, 0.0111], Holm p 0.822, not supported; gray-world -0.0284 [-0.0478, -0.0031], p 0.0535, not supported; fusion -0.0255 [-0.0446, -0.0019], p 0.0875, not supported.
- mAP@50-95 deltas: CLAHE -0.0021, p 0.822, not supported; gray-world -0.0229, p 0.0365, supported degradation; fusion -0.0251, p 0.0365, supported degradation.
- "Supported" requires CI excluding zero and Holm p < 0.05 across three methods x two metrics. No enhancement improvement was supported.
- Artifacts: `outputs/enhancement_accuracy_test_clean.json`, `outputs/bootstrap_uncertainty.json`.

## OBSEA independent-domain check (completed)

- Downloaded with approval from Zenodo record `14888440`, DOI `10.5281/zenodo.14888440`, CC BY 4.0; zip 3,784,147,485 bytes; MD5 `80daf33935f424e66fb35e31ef3b1780`.
- Archive: `external_data/obsea/obsea_split_YOLO.zip`; extracted root: `external_data/obsea/extracted/23sp_4120img_34945annots_2688res`.
- 2,884 train / 618 valid / 618 test images; 34,945 rows; 23 classes; 21 fish taxa; non-fish `Diver`, `Octopus vulgaris`.
- Test: 618 images; 5,493 rows; 5,373 fish instances; 120 non-fish; 548 images with fish; 70 backgrounds. Derived test: 5,369 instances after removing 4 duplicate boxes; all images retained; hardlinked at `outputs/derived/obsea_fish_test`.
- Zero-shot, class-agnostic, no tuning: P 0.124, R 0.094, mAP@50 0.036, mAP@50-95 0.017, mAP@75 0.013; 29.7 ms/image CPU.
- This is not 13-class species accuracy and not universal generalization. Predictions on excluded Diver/Octopus count as false positives.
- Artifact: `outputs/obsea_evaluation.json`; script: `scratch/run_obsea_evaluation.py`.

## RT-DETR status

- `models/rtdetr-l.pt`: COCO-pretrained, 80 classes, 32,970,476 parameters, 66,511,432 bytes, SHA-256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`.
- `outputs/latency_yolov8_vs_rtdetr.json` remains timing-only: YOLOv8 60.74 ms, RT-DETR 706.32 ms mean inference on 30 CPU images.
- Fair-comparison plan: `rtdetr-finetune-plan.md`.
- Opt-in script: `scratch/run_rtdetr_finetune.py`. Default invocation does nothing; `--pilot`, `--train`, and `--evaluate` require `--execute`.
- Prepared data: `outputs/derived/rtdetr_fish_train` (6,834 images/labels), `outputs/derived/rtdetr_fish_valid` (698/698), `outputs/derived/rtdetr_fish/data.yaml`.
- No RT-DETR install, training, or fish-accuracy evaluation was run. Do not claim a comparison.

## App and API

- `/api/metrics` exposes `baseline`, `heldout_evaluation`, `bootstrap_uncertainty`, `obsea_evaluation`, `enhancement_accuracy`, `latency_benchmark`, `enhancement_audit`, `enhancement_benchmark`, and provenance notes.
- Home and Research pages were updated with bootstrap intervals, OBSEA results/limitations, RT-DETR status, leakage counts, and real-video limitations.
- Demo page identifies RT-DETR as a COCO-pretrained timing baseline; video tracking is labeled as synthetic-smoke-test coverage only.
- API tests assert the new metrics keys and OBSEA license.

## Remaining blockers

1. Real-video tracking was run on the approved CC BY 4.0 DIGI4ECO OBSEA clip in `.venv-video`. The video decodes to 148 frames while annotations span 150; 14 boxes beyond the video were excluded, leaving 1,327 reference boxes. Ground truth has 12 tracks and 6 gate crossings. At confidence 0.25, inference yielded zero detections/IDs/counts. At 0.001 it yielded 450 raw low-confidence detections (mean confidence 0.0041), all below ByteTrack's 0.1 low threshold, thus still zero IDs/counts. Count error was six (100%) in both runs. The run validates pipeline execution and exposes a model/tracker failure; it does not validate persistent IDs. See `video-tracking-validation.md` and [`../external_data/digi4eco/README.md`](../external_data/digi4eco/README.md).
2. Fair RT-DETR accuracy: requires approval to create a separate CUDA 13.x environment, pass a GPU tensor gate, run a one-epoch pilot, then approve a full run.
3. Universal cross-domain claim: only OBSEA was evaluated; another labeled domain would be needed.
4. Source-independent local test: leakage-filtered subset is not proof of independence.
5. Git/secrets: no `.git`; no tracked-history audit or secret scan is possible here.
6. Research manuscript: `research-paper.md` is a draft based on saved evidence, explicitly not peer reviewed or published.

## Before taking a task

1. Check `AGENTS.md`, `README.md`, and the exact target files.
2. Keep new findings separate from historical saved results; record checkpoint hash, dataset, device, settings, and output path.
3. Preserve user artifacts and limit changes to the request.
4. Coordinate file ownership; do not assume Git isolation.
5. Report commands and evidence precisely, including failures and limitations.

## Suggested next steps

1. Obtain approval for a separate RT-DETR environment and run the GPU gate plus one-epoch pilot.
2. Do not tune against this same evaluation clip. A future tracking study needs a separate development clip for threshold/tracker selection and a held-out reference clip for evaluation. For this checkpoint on the current clip, tracking/counting failed.
3. Decide whether to download/fine-tune a matched YOLOv8n control for Tier 1 fairness.
4. If a second external labeled dataset is added, evaluate it without mixing metrics with OBSEA or local results.
