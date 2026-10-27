# DeepSea AI - Underwater Fish Detection Audit

This repository contains the original YOLOv8 fish detector plus enhancement, inference benchmarking, image demonstration, and video tracking code. This README records what the available artifacts support; it does not treat demonstration outputs as validated research results.

## Working with different coding agents

Project guidance is centralized in [`AGENTS.md`](AGENTS.md). Codex and Antigravity can use it directly; [`CLAUDE.md`](CLAUDE.md) and [`GEMINI.md`](GEMINI.md) import the same shared guidance. The current project state is summarized in [`docs/agent-handoff.md`](docs/agent-handoff.md). Keep provider-specific files short and put shared rules in `AGENTS.md` to avoid drift.

See [`docs/project-map.md`](docs/project-map.md) for the folder map, file-naming conventions, and rules for preserving experiment artifacts. The evidence-based manuscript draft is [`docs/research-paper.md`](docs/research-paper.md); it is not peer reviewed or published.

Multiple agents can work on the project, but this local workspace has no `.git` metadata, so Git worktrees and branch-based isolation are unavailable. Coordinate file ownership and avoid simultaneous edits. Reopen the project from a valid Git clone to use worktrees safely.

## Current verification (2026-10-28)

- The recorded OpenCode run on 2026-10-28 used Python 3.11.9 with `torch 2.14.1+cu126`, `ultralytics 8.4.190`, `opencv-python 5.0.0`, `fastapi 0.162.0`, and `httpx 0.28.1`.
- That OpenCode run collected and passed **57 tests in 11.30 s**. It included eight API integration tests and 14 uncertainty tests, with Starlette/httpx deprecation, pytest fixture, and PyTorch sm_120 warnings. Passing tests validate code paths and schemas, not fish-detection accuracy.
- The OpenCode frontend run passed `npm run build` with Vite 8.0.2 in 715 ms. `npm run check:routes` rendered `/`, `/demo`, `/research`, and `/contact` without errors. The route check is an SSR smoke test, not a browser end-to-end test.
- In the current audit shell, `.venv\Scripts\python.exe` cannot launch because its configured Python 3.11.9 executable is missing. Available system Python is 3.14.7 and lacks `cv2`, `torch`, and `ultralytics`; the recorded Python results above were not reproduced in this shell. No dependencies were installed.
- The current system-Python attempt `python -m pytest tests -v` collected 14 items but stopped with 2 collection errors (`ModuleNotFoundError: No module named 'cv2'`); no tests ran in that attempt. This is separate from, and does not erase, the recorded OpenCode pass above.
- GPU probe: `torch.cuda.is_available()` is `True` for the RTX 5050 Laptop GPU (8,151 MiB, compute capability 12.0, driver 610.74), but `torch 2.14.1+cu126` has no sm_120 kernels. A minimal GPU tensor operation fails with `cudaErrorNoKernelImageForDevice`, so `get_optimal_device()` falls back to CPU. The installed build suggests CUDA 13.0/13.2/13.4 wheels. No GPU inference, training, or PyTorch reinstall was performed.
- An earlier attempt to launch the configured Python 3.11 executable failed; system Python 3.14.7 stopped collection because `cv2` was missing. The current `.venv` result above supersedes those failures.
- No `.git` directory is present, so Git status/history and tracked-file provenance cannot be established here.
- The approved CC BY 4.0 OBSEA image dataset and DIGI4ECO tracking dataset are present. DIGI4ECO was downloaded with user approval, checksum-verified, and only its OBSEA video and annotations were extracted. An isolated CPU environment was approved for the real-video run; the tested checkpoint produced no track IDs. No model was trained.

## Dataset and evaluation protocol

- Local dataset: 8,229 images (6,834 train / 698 valid / 697 test), 640x640 RGB, 13 classes.
- `models/best.pt`: custom YOLOv8n fish checkpoint, 5.96 MB, SHA-256 `af229161766094f3f002ff301b68001f6bec84492fe4f43c41c9f0838d1a5d23`.
- Labels mix 5-field YOLO boxes and longer polygon rows. Polygons were converted to enclosing boxes for evaluation. This changes the supervision geometry for 84 validation and 29 test instances.
- Derived YAML files under `outputs/derived/` are used because `Dataset/data.yaml` split paths do not resolve in the current layout.
- Leakage audit:
  - 34 test images exact-match a train image; 33 distinct hashes.
  - 212 test images share a train source stem; 202 distinct prefixes.
  - 70 test images share a valid source stem; 69 distinct prefixes.
  - `test_clean` excludes 298 test images and is conservative, not proof of source independence.
- `test_sample` contains 40 images and no labels; 39 of 40 are exact duplicates of Dataset images. It is not an independent cross-domain dataset.

## Held-out evaluation

The saved checkpoint was evaluated with Ultralytics `YOLO.val` on CPU using `imgsz=640`, `conf=0.001`, `iou=0.6`, and `workers=0`.

| Split | Images | Instances | Precision | Recall | mAP@50 | 95% CI | mAP@50-95 | 95% CI | Interpretation |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| `valid_full` | 698 | 1,193 | 0.880 | 0.875 | 0.910 | 0.886-0.939 | 0.658 | 0.637-0.698 | Used during training/model selection; not independent. AngelFish has 0 instances. |
| `test_full` | 697 | 1,129 | 0.816 | 0.804 | 0.840 | 0.826-0.874 | 0.622 | 0.604-0.670 | Not fully independent: 34 exact train matches, 212 train-stem overlaps, 70 valid-stem overlaps. |
| `test_clean` | 399 | 639 | 0.809 | 0.803 | 0.841 | 0.823-0.880 | 0.636 | 0.615-0.694 | Leakage-filtered subset (298 excluded). Conservative, but not source-independent. |

- Intervals are image-level percentile bootstrap intervals (2,000 resamples, seed `20261027`). Source-stem cluster intervals are similar, for example `test_clean` mAP@50 0.824-0.880 and mAP@50-95 0.617-0.693.
- Overall mAP intervals are conditioned on every evaluated class having at least one ground-truth instance in the resample. Finite resamples were 1,908/2,000 for `test_full`, 1,644/2,000 for `test_clean`, and 1,997/2,000 for `valid_full`. The intervals do not fix leakage or prove source independence.
- Point estimates reproduce the saved validation metrics within `max|delta| < 4.6e-7`.
- The historical notebook validation log remains precision **0.865**, recall **0.874**, mAP@50 **0.908**, and mAP@50-95 **0.666** (12 of 13 classes; AngelFish had 0 validation instances). The local `valid_full` re-evaluation is close but not identical.
- AngelFish is absent from validation but present in test: 227 instances on `test_full` (AP@50 0.953, AP@50-95 0.734) and 173 on `test_clean` (0.951 / 0.741).
- The artifact flags ButterflyFish (2), ZebraFish (4), and RibbonedSweetlips (8) as low-support in `test_clean`; Gourami (10), MorishIdol (14), and GoldFish (20) are also small. YellowCichlid is the weakest well-supported class (AP@50 0.672, AP@50-95 0.282 on both test sets). PlatyFish is also weaker (0.771 / 0.526 on `test_full`; 0.729 / 0.512 on `test_clean`).
- The near-identical overall mAP@50 values of `test_full` and `test_clean` are not evidence that leakage is harmless. Class composition differs, and `test_clean` still shares source provenance with training data.
- Artifacts: `outputs/heldout_yolov8_metrics.json`, `outputs/bootstrap_uncertainty.json`, `outputs/uncertainty_stats/`. Scripts: `scratch/run_heldout_evaluation.py`, `scratch/run_bootstrap_uncertainty.py`, `src/evaluation/uncertainty.py`.

## Enhancement accuracy and uncertainty

The same checkpoint and 399-image `test_clean` subset (639 instances) were used. `raw_original` hardlinks the original JPEGs; all other rows were written as JPEG quality 95, including the `raw_jpeg95` re-encode control. Enhancement latency was measured on CPU after warmup.

| Method | mAP@50 | 95% CI | mAP@50-95 | 95% CI | Precision | Recall | Mean enhancement ms |
|---|---:|---|---:|---|---:|---:|---:|
| `raw_original` | 0.841 | 0.823-0.880 | 0.636 | 0.615-0.694 | 0.809 | 0.803 | - |
| `raw_jpeg95` | 0.841 | 0.823-0.879 | 0.636 | 0.615-0.695 | 0.792 | 0.809 | - |
| `clahe` | 0.846 | 0.829-0.881 | 0.634 | 0.614-0.687 | 0.792 | 0.813 | 11.9 |
| `gray_world` | 0.812 | 0.787-0.864 | 0.613 | 0.589-0.675 | 0.796 | 0.745 | 10.5 |
| `fusion` | 0.815 | 0.790-0.865 | 0.611 | 0.587-0.673 | 0.791 | 0.760 | 15.1 |

Paired image-bootstrap deltas, left minus right:

| Comparison | Metric | Delta | 95% CI | Holm p | Supported |
|---|---|---:|---|---:|---|
| CLAHE vs JPEG-95 control | mAP@50 | +0.0052 | -0.0028 to +0.0111 | 0.822 | No |
| CLAHE vs JPEG-95 control | mAP@50-95 | -0.0021 | -0.0143 to +0.0050 | 0.822 | No |
| Gray-world vs JPEG-95 control | mAP@50 | -0.0284 | -0.0478 to -0.0031 | 0.0535 | No |
| Gray-world vs JPEG-95 control | mAP@50-95 | -0.0229 | -0.0407 to -0.0050 | 0.0365 | Yes, degradation |
| Fusion vs JPEG-95 control | mAP@50 | -0.0255 | -0.0446 to -0.0019 | 0.0875 | No |
| Fusion vs JPEG-95 control | mAP@50-95 | -0.0251 | -0.0428 to -0.0067 | 0.0365 | Yes, degradation |

"Supported" requires the paired interval to exclude zero and the Holm-adjusted p value to be below 0.05 across three methods and two metrics. CLAHE was approximately neutral. Gray-world and fusion did not improve accuracy; their mAP@50-95 degradations remained supported after correction. The comparison uses one checkpoint, one leakage-filtered subset, and CPU inference. Artifact: `outputs/enhancement_accuracy_test_clean.json`; paired summaries are in `outputs/bootstrap_uncertainty.json`.

The older 40-image `test_sample` audit remains a counts/confidence/timing observation, not an accuracy measurement. No labels exist for those images.

## OBSEA independent-domain check (new, 2026-10-28)

OBSEA is an external Mediterranean observatory fish dataset downloaded from Zenodo record `14888440`, DOI `10.5281/zenodo.14888440`, under CC BY 4.0. The archive is 3,784,147,485 bytes and was checksum-verified with MD5 `80daf33935f424e66fb35e31ef3b1780` before extraction.

- Archive: `external_data/obsea/obsea_split_YOLO.zip`.
- Extracted root: `external_data/obsea/extracted/23sp_4120img_34945annots_2688res`.
- Counts: 2,884 train / 618 valid / 618 test images; 34,945 annotation rows; 23 classes; 21 fish taxa plus `Diver` and `Octopus vulgaris`.
- Test split: 618 images, 5,493 annotation rows, 5,373 fish instances, 120 non-fish instances, 548 images with fish, 70 background images.
- Derived class-agnostic test set: all 618 images retained; non-fish annotations excluded; identical fish boxes deduplicated (4 duplicate rows removed); 5,369 fish instances evaluated.
- Protocol: class-agnostic ground truth and predictions, `single_cls=True`, `agnostic_nms=True`, `imgsz=640`, `conf=0.001`, `iou=0.6`, `max_det=300`, CPU, no OBSEA tuning or model selection.

| Metric | Value |
|---|---:|
| Precision | 0.124 |
| Recall | 0.094 |
| mAP@50 | 0.036 |
| mAP@50-95 | 0.017 |
| mAP@75 | 0.013 |
| Mean inference ms/image | 29.7 |

- This measures class-agnostic fish detection under domain shift, not 13-class species accuracy. Local and OBSEA taxonomies do not match.
- Predictions on excluded Diver and Octopus objects remain false positives because every prediction is treated as fish.
- The low result is evidence of substantial domain shift on one external dataset. It does not establish universal cross-domain performance.
- Artifact: `outputs/obsea_evaluation.json`; derived data: `outputs/derived/obsea_fish_test`; script: `scratch/run_obsea_evaluation.py`.

## Real-video tracking run: zero detections at the primary threshold

The user-approved [DIGI4ECO tracking dataset](https://doi.org/10.5281/zenodo.20701225) is licensed CC BY 4.0. Its archive was verified against Zenodo's MD5. Only the OBSEA video and CVAT XML were extracted to `external_data/digi4eco/extracted/OBSEA/`; the annotation file contains 12 tracks and 1,384 boxes over 150 frames. Ground-truth-only analysis found six crossings at the default center gate; the parser and saved reference are `scratch/analyze_cvat_gate_reference.py` and `outputs/digi4eco_obsea_gate_reference.json`. The archive and extracted data are ignored by Git. The clip is from the OBSEA domain and is not a second independent geography.

The approved isolated CPU environment (`.venv-video`) processed the clip. It decoded 148 frames while CVAT labels cover 150; 14 visible, non-occluded boxes past the decoded video were excluded, leaving 1,327 reference boxes and six gate crossings. At confidence 0.25 the model emitted zero detections and IDs. At confidence 0.001 it emitted 450 boxes (mean confidence 0.0041), all below ByteTrack's 0.1 low threshold, so still no IDs or crossings. Both runs missed all six crossings (100% absolute count error; event F1 0). This is a completed negative pipeline evaluation, not successful persistent-ID validation. Results: `outputs/digi4eco_obsea_tracking_primary_v2.json` and `outputs/digi4eco_obsea_tracking_conf001_thresholds.json`; protocol: [`docs/video-tracking-validation.md`](docs/video-tracking-validation.md).

## Latency-only comparison

This is a timing comparison, not an accuracy comparison. The RT-DETR checkpoint is pretrained on COCO and has not been fine-tuned on the fish dataset, so its detections are not fish-species predictions.

| Model | Checkpoint | Classes | Mean inference ms | Mean wall ms | Mean detections/image |
|---|---|---:|---:|---:|---:|
| YOLOv8n fish | `models/best.pt` | 13 | 60.74 | 68.65 | 1.2 |
| RT-DETR-L COCO | `models/rtdetr-l.pt` | 80 | 706.32 | 715.27 | 3.3 |

- Protocol: 30 `test_clean` images, 3 warmup images, `imgsz=640`, `conf=0.25`, batch size 1, CPU, Ultralytics `predict`.
- `models/rtdetr-l.pt`: 32,970,476 parameters, 66,511,432 bytes, SHA-256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`.
- A fair RT-DETR accuracy comparison requires a compatible GPU environment and fish fine-tuning. The plan is in `docs/rtdetr-finetune-plan.md`; the opt-in script is `scratch/run_rtdetr_finetune.py`. No training or fish-accuracy evaluation has been executed.
- Artifact: `outputs/latency_yolov8_vs_rtdetr.json`; script: `scratch/run_latency_benchmark.py`.

## VERIFIED

- `models/best.pt` is the custom fish YOLO checkpoint (5.96 MB, SHA-256 `af229161766094f3f002ff301b68001f6bec84492fe4f43c41c9f0838d1a5d23`). Its 13 class names match `Dataset/data.yaml`. The original notebook log reports 10 epochs and validation precision 0.865, recall 0.874, mAP@50 0.908, and mAP@50-95 0.666. Those are notebook log values, not a newly run evaluation or held-out result.
- `models/rtdetr-l.pt` exists (66,511,432 bytes) and loads as an RT-DETR-L checkpoint. It is pretrained on COCO, not fish fine-tuned.
- Dataset audit: 8,229 images (6,834 train / 698 valid / 697 test), all 640x640 RGB and readable. After polygon conversion the instance totals are 14,144 train / 1,193 valid / 1,129 test. All 13 classes are present; AngelFish has 0 validation instances. The Roboflow README's 8,242-image count does not match the 8,229 files present.
- `src/data/validator.py` parses 5-field boxes and >5-field polygons, converts polygons to enclosing boxes, and reports `box_rows`, `polygon_rows`, and `converted_polygon_boxes`; focused tests cover valid polygons, malformed rows, out-of-bounds polygons, and class IDs. On the local dataset it reports 14,144 / 1,193 / 1,129 instances with 0 invalid rows and 0 out-of-bounds boxes.
- `test_sample` contains 40 images and no labels; 39 of 40 are exact duplicates of Dataset images. It is not independent cross-domain evidence.
- `scratch/run_enhancement_audit.py` evaluates a fixed, alphabetically sorted set of 40 `test_sample/*.jpg` images with the same YOLO checkpoint, confidence threshold 0.25, and four enhancement modes. Saved counts/confidence/latency are qualitative inference observations only.

| Method | Images with detections | Detection boxes | Mean box confidence | Mean preprocess ms | Mean inference ms | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 40/40 | 61 | 0.6995 | 7.55 | 150.06 | 157.61 |
| CLAHE | 36/40 | 54 | 0.7151 | 20.66 | 56.93 | 77.60 |
| Gray-world | 39/40 | 65 | 0.6823 | 16.54 | 67.56 | 84.09 |
| Fusion | 40/40 | 63 | 0.6602 | 27.26 | 59.09 | 86.35 |

Timings are from the saved CPU run and may vary with machine load/cache. Higher counts or confidence do not mean better detection accuracy; false positives and missed fish cannot be assessed without annotations. `outputs/enhancement_benchmark.json` is an earlier output with materially different timing values; `outputs/enhancement_audit.json` is the newer per-image record and is the source of the table above.

- OBSEA download, MD5 verification, extraction, audit, derived-data preparation, and the zero-shot evaluation all completed. The recorded metrics are preliminary external evidence, not a species result.
- The bootstrap script reproduced the saved held-out point estimates and wrote image/cluster intervals. The uncertainty tests pass, but they validate matching/bootstrap logic only.
- The RT-DETR data-preparation path was executed: `outputs/derived/rtdetr_fish_train` has 6,834 images/labels, `outputs/derived/rtdetr_fish_valid` has 698/698, and `outputs/derived/rtdetr_fish/data.yaml` has 13 classes. No RT-DETR training or evaluation was run.
- The FastAPI `/api/metrics` response now includes `bootstrap_uncertainty` and `obsea_evaluation` with provenance notes. API tests assert the new keys and OBSEA license.

## PRELIMINARY

- The held-out evaluation and bootstrap intervals are preliminary because `test_full` is not fully independent and `test_clean` is only leakage-filtered. The intervals are conditional on class presence and are not external validation.
- The OBSEA result is preliminary: one external dataset, one checkpoint, zero-shot, class-agnostic, and no species mapping. It should not be merged with local 13-class metrics.
- The enhancement comparison is preliminary: one checkpoint, one 399-image subset, CPU inference, and low-support classes. Only the mAP@50-95 degradations for gray-world and fusion survived Holm correction; no enhancement method showed a supported improvement.
- `scratch/run_model_benchmark.py` performs forward inference with `models/best.pt` and pretrained `models/rtdetr-l.pt` on a single all-zero image, with warmup and ten timed runs. It does not use a fish evaluation set or compare accuracy. The saved CPU timings are YOLOv8n-Fish: 3,013,383 parameters, 5.96 MB, 93.50 ms (10.69 FPS); RT-DETR-L: 32,970,476 parameters, 63.43 MB, 912.77 ms (1.10 FPS). This is a latency/resource benchmark only.
- `scratch/run_cross_domain.py` labels `test_sample` as a target domain, but `src/evaluation/cross_domain.py` simply runs inference on those same 40 sample images. There is no independent target dataset or ground truth in that artifact. It is same-distribution qualitative inference, not cross-domain generalization evidence.
- `scratch/run_video_tracking_smoketest.py` creates synthetic colored rectangles, not fish, and checks only that processing returns the expected fields and frame count. It does not validate fish detection or real-video ID accuracy.
- A real annotated video run is saved, but the tested checkpoint produced no IDs or crossings. Tracking/counting remains unsuccessful for this checkpoint and configuration; see the tracking protocol and JSON artifacts.
- The frontend route check is an SSR smoke test, not a browser end-to-end test. No visual regression or accessibility audit was performed.

## NOT YET COMPLETED

- **Fair RT-DETR accuracy comparison:** only the COCO-pretrained checkpoint exists. The plan and opt-in script are present, and derived training data is prepared, but no compatible GPU environment was installed and no fish fine-tuning or evaluation was run. The saved RT-DETR result is timing-only.
- **GPU/PyTorch:** the machine reports an RTX 5050 Laptop GPU (compute capability 12.0 / Blackwell), 8,151 MiB VRAM, and driver 610.74. The installed `torch 2.14.1+cu126` exposes CUDA but has no sm_120 kernels: a minimal GPU operation fails with `cudaErrorNoKernelImageForDevice`, so the application runs on CPU. A compatible PyTorch installation is a candidate next step, not a tested environment.
- **Video:** real video and ground-truth annotations were processed, but the checkpoint produced zero track IDs at confidence 0.25 and 0.001. It missed all six reference crossings. No successful persistent-ID result or MOTA/IDF1/HOTA score is available. Current evidence and failure interpretation are in `docs/video-tracking-validation.md`.
- **Universal independent-domain validation:** OBSEA is one external Mediterranean dataset. No second labeled domain, habitat, or geography was evaluated. Do not describe OBSEA as proof of universal generalization.
- **Source independence:** `test_clean` is leakage-filtered but still shares source-level provenance with training data. The source-stem cluster bootstrap is a sensitivity analysis, not proof of independence.
- **Git/secrets:** this workspace has no `.git` directory, so `git status`, tracked-file checks, and history inspection could not be performed. No secret values were printed in this pass, but the absence of secrets in tracked history is not established. If a credential was ever exposed, revoke it and inspect the actual clone separately.
- **Research PDFs:** `client/public/research/RE_Paper.pdf`, `client/public/research/sample_paper.pdf`, and `client/src/assets/RE_Paper.pdf` are preserved artifacts. No evidence in this workspace establishes peer review or publication.
- **Environment reproducibility:** root and server dependency files use broad lower bounds and do not pin a reproducible environment. No Python lockfile is present.
- **Statistical scope:** confidence intervals are implemented for the saved local evaluations, but not for OBSEA, RT-DETR, real-video tracking, or enhancement latency. No multiple-comparison correction was applied outside the enhancement mAP comparisons.

## Experiment artifacts

- `scratch/eval_common.py` - shared label parsing/conversion, derived-dataset preparation, and Ultralytics evaluation helpers.
- `scratch/run_dataset_audit.py`, `outputs/dataset_audit.json` - dataset counts, formats, duplicate hashes, source-stem overlaps, image integrity, checkpoint compatibility, and `test_sample` audit.
- `scratch/run_heldout_evaluation.py`, `outputs/heldout_yolov8_metrics.json` - `test_full`, `test_clean`, and `valid_full` re-evaluation.
- `src/evaluation/uncertainty.py`, `scratch/run_bootstrap_uncertainty.py`, `tests/test_uncertainty.py` - image/cluster bootstrap, paired deltas, Holm correction, and tests.
- `outputs/bootstrap_uncertainty.json`, `outputs/uncertainty_stats/` - saved intervals and exact per-image validation stats.
- `scratch/run_enhancement_accuracy.py`, `outputs/enhancement_accuracy_test_clean.json` - labeled enhancement comparison on `test_clean`.
- `scratch/run_latency_benchmark.py`, `outputs/latency_yolov8_vs_rtdetr.json` - 30-image timing-only YOLOv8 vs pretrained RT-DETR comparison.
- `scratch/run_obsea_evaluation.py`, `outputs/obsea_evaluation.json` - zero-shot class-agnostic OBSEA evaluation.
- `external_data/obsea/obsea_split_YOLO.zip`, `external_data/obsea/extracted/23sp_4120img_34945annots_2688res` - verified external dataset and extracted source.
- `outputs/derived/obsea_fish_test` - class-agnostic derived test set.
- `docs/rtdetr-finetune-plan.md`, `scratch/run_rtdetr_finetune.py`, `outputs/derived/rtdetr_fish*` - opt-in RT-DETR plan, script, and prepared data. No training result exists.
- `scratch/run_model_benchmark.py`, `outputs/model_benchmark_comparison.json` - zero-image latency/resource benchmark.
- `scratch/run_enhancement_audit.py`, `outputs/enhancement_audit.json` - fixed 40-image enhancement inference run.
- `outputs/enhancement_benchmark.json` - earlier aggregate enhancement run; do not mix its timings with the newer audit.
- `scratch/run_cross_domain.py`, `outputs/cross_domain_evaluation.json` - same-distribution `test_sample` inference, not OOD evaluation.
- `scratch/run_video_tracking_smoketest.py` - synthetic-video smoke test only.
- `docs/video-tracking-validation.md` - real-video protocol and executed negative result; the tested checkpoint produced no IDs or crossings at the two reported confidence settings.

## Run the project

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Current result: 57 passed in 11.30 s. The suite includes API integration tests; the video test uses synthetic frames.

From `client/`:

```powershell
npm run build
npm run check:routes
```

The current build and route checks pass. The route check is not a browser end-to-end test.

The API reads optional `ALLOW_ORIGINS` (comma-separated; default `*`), reports the actually used model plus fallback status, validates upload size/threshold parameters, and exposes the historical baseline plus held-out, bootstrap, OBSEA, enhancement, and latency artifacts from `/api/metrics` with provenance notes.

RT-DETR data preparation is available without training:

```powershell
.\.venv\Scripts\python.exe scratch\run_rtdetr_finetune.py --prepare-data
```

Do not run `--train`, `--pilot`, or `--evaluate` without reviewing `docs/rtdetr-finetune-plan.md` and obtaining approval for the separate GPU environment.

No large datasets beyond the approved OBSEA download, no training jobs, and no package installations were started in this pass. Existing checkpoints, notebooks, dataset files, `test_sample` images, PDFs, and saved JSON outputs were preserved. New derived datasets and evaluation artifacts were written under `outputs/`, and new scripts/docs under `scratch/` and `docs/`.
