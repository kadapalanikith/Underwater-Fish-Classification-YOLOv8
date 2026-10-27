# Shared repository instructions

This file is the canonical project guidance for any coding agent working in this repository. Read it together with `readme.md` and `docs/agent-handoff.md` before making changes. Provider-specific entry files should point here rather than maintain competing project rules.

## Project facts and evidence

- This is an underwater fish detection project built around a saved YOLOv8 checkpoint, a FastAPI server, and a React/Vite client. Enhancement, RT-DETR benchmarking, external-domain evaluation, and video tracking code also exist, but their evidence is limited as described in the README and handoff.
- Treat `outputs/` as recorded artifacts, not automatically as reproducible or verified results. Preserve each output's provenance and limitations. Do not merge metrics from separate runs.
- The original training metrics are validation-log values. Do not describe them as independent test performance; the original validation logs have no AngelFish validation instances.
- A separate held-out re-evaluation is recorded in `outputs/heldout_yolov8_metrics.json`. `test_clean` (399 leakage-filtered images / 639 instances) reached mAP@50 0.841 and mAP@50-95 0.636; `test_full` (697 / 1,129) reached 0.840 / 0.622; `valid_full` (698 / 1,193) reached 0.910 / 0.658. `test_full` is not fully independent and `test_clean` is a conservative leakage-filtered subset, not proof of source independence. Polygons were converted to enclosing boxes. Keep these metrics separate from the historical notebook log.
- Bootstrap intervals are recorded in `outputs/bootstrap_uncertainty.json`. The primary intervals are image-level percentile intervals with 2,000 resamples, seed `20261027`; source-stem cluster intervals are a sensitivity check. Overall mAP intervals require every evaluated class to be present. They do not fix leakage or prove source independence.
- A labeled enhancement comparison on `test_clean` is recorded in `outputs/enhancement_accuracy_test_clean.json`: CLAHE was approximately neutral; only the mAP@50-95 degradations for gray-world and fusion were supported after Holm correction. No enhancement improvement was supported.
- An approved external-domain download is recorded under `external_data/obsea/`. OBSEA is CC BY 4.0, DOI `10.5281/zenodo.14888440`, with a checksum-verified archive. The zero-shot result in `outputs/obsea_evaluation.json` is class-agnostic fish detection, not 13-class species accuracy. It is one Mediterranean dataset, not universal cross-domain proof.
- A user-approved CC BY 4.0 DIGI4ECO archive is under `external_data/digi4eco/` (DOI `10.5281/zenodo.20701225`). Only its OBSEA video and CVAT annotations were extracted. Ground truth has 12 tracks and 1,384 boxes through frame 149; the video decodes to 148 frames, so 14 trailing visible boxes were excluded. In isolated `.venv-video`, confidence 0.25 produced zero detections; confidence 0.001 produced 450 raw boxes, all below ByteTrack's 0.1 low threshold. Both runs produced zero IDs/counts against six reference crossings. This is a negative tracker result, not tracking validation success; see `docs/video-tracking-validation.md`.
- `outputs/latency_yolov8_vs_rtdetr.json` is a timing-only comparison. `models/rtdetr-l.pt` is a COCO-pretrained 80-class checkpoint, not fish fine-tuned; its detections are not fish-species predictions. No RT-DETR fish training or accuracy evaluation has been run.
- `docs/rtdetr-finetune-plan.md` and `scratch/run_rtdetr_finetune.py` define an opt-in RT-DETR plan. The script does nothing by default; pilot, train, and evaluate require `--execute`. Do not run them without approval.
- `test_sample/` is not an independent cross-domain dataset and has no ground-truth annotations in this workspace. It is 39/40 exact duplicates of Dataset images.
- The tracking smoke-test script uses synthetic rectangles. A separate real-video evaluation ran on DIGI4ECO OBSEA, but the detector produced no IDs or crossings at confidence 0.25 or 0.001; see the README and video protocol. Do not describe that negative result as successful tracking validation.
- PDFs under `client/public/research/` are preserved artifacts, not evidence of publication or peer review.
- Dataset validator: `src/data/validator.py` parses 5-field detection boxes and >5-field polygons, converts polygons to enclosing boxes, and reports `box_rows`, `polygon_rows`, and `converted_polygon_boxes`. Focused tests cover valid polygons, malformed rows, out-of-bounds polygons, and class IDs. The evaluation pipeline still uses `scratch/eval_common.py` for derived dataset preparation.

## Change practices

- Inspect files and current behavior before editing. Prefer small, reversible changes that preserve existing user work.
- Do not delete, overwrite, rename, or regenerate checkpoints, notebooks, datasets, experiment outputs, or research PDFs unless the user explicitly requests it.
- Do not invent measurements, citations, annotation availability, completed experiments, or runtime support. Mark unknowns as unknown and distinguish code inspection from execution.
- Do not display or commit credentials. Keep secrets out of notebooks, outputs, logs, and docs. Use environment variables or a user-managed secret store.
- Do not download datasets, train models, install heavyweight dependencies, or push/publish changes without the user's approval. The OBSEA download was explicitly approved; that approval does not extend to other datasets or RT-DETR installation/training.
- Do not claim tests passed unless they ran successfully in the current environment. Report collection failures, skipped tests, and commands exactly.
- Before changing shared files during parallel work, coordinate ownership with other agents. Use separate worktrees/branches where available. If working from a source export without `.git`, do not assume Git-based isolation or claim a Git status; verify the actual checkout first.
- Keep provider-specific instructions in thin adapter files; put durable project rules here.

## Repository map

- `server/main.py` - FastAPI API
- `client/src/` - React/Vite user interface
- `src/preprocessing/` - image enhancement
- `src/models/` - model benchmarking
- `src/evaluation/` - evaluation and uncertainty helpers
- `src/tracking/` - video tracking/counting
- `src/data/` - dataset validation
- `scratch/` - experiment and smoke-test scripts
- `external_data/` - approved external data and provenance (ignored by Git; preserve licenses and checksums)
- `outputs/` - recorded JSON artifacts and derived datasets
- `docs/` - audit handoff, RT-DETR plan, video-tracking protocol, project map, and research-paper draft
- `tests/` - Python tests, including FastAPI integration tests
- `notebooks/` - original training notebooks

## Commands and environment

From the repository root:

```sh
python -m pytest tests -v
```

Current OpenCode run on 2026-10-28: the local `.venv` (Python 3.11.9) collected and passed **57 tests in 11.30 s** with `torch 2.14.1+cu126`, `ultralytics 8.4.190`, `opencv-python 5.0.0`, `fastapi 0.162.0`, and `httpx 0.28.1`. The run emitted Starlette/httpx deprecation, pytest fixture deprecation, and PyTorch sm_120 warnings. The suite includes API integration tests; the video test uses synthetic frames. Passing tests validate code paths and schemas, not fish-detection accuracy. Re-run successfully before claiming current pass status; do not install dependencies without user approval. No Python environment lock file is present.

The current audit created `.venv-video` with explicit user approval for real-video processing. In the Git checkout, `YOLO_CONFIG_DIR=.cache/ultralytics-video .venv-video\Scripts\python.exe -m pytest tests -v --ignore=tests/test_api.py` passed **52 tests in 7.05 s** (one pytest fixture deprecation warning). The full suite could not collect because FastAPI is absent from this video-only environment. The original `.venv` still points to a missing Python 3.11 executable; system Python 3.14.7 previously stopped collection because `cv2` was missing. Do not install additional dependencies without user approval.

Frontend dependencies are in `client/package.json` with `client/package-lock.json`. If `client/node_modules` is present, do not reinstall packages unnecessarily. Check the available scripts in `client/package.json` before running a build or lint command. From `client/`, `npm run build` and `npm run check:routes` both passed in the current recheck with Vite 8.0.2; no lint script is defined. The route check is an SSR smoke test, not a browser end-to-end test.

For GPU work, inspect the installed Python/PyTorch/CUDA/GPU combination first. The machine has an RTX 5050 Laptop GPU with about 8 GiB VRAM. The installed `torch 2.14.1+cu126` reports CUDA as available but fails a minimal GPU tensor operation on compute capability 12.0 (`cudaErrorNoKernelImageForDevice`); `get_optimal_device()` consequently falls back to CPU. Do not claim GPU inference, and do not reinstall PyTorch without user approval. Do not infer PyTorch CUDA compatibility from `nvidia-smi` alone. If a separate RT-DETR environment is approved, verify a real GPU tensor operation before any training.

## Completion notes

When handing work back, summarize changed files, commands actually run, their results, and any remaining blockers. For experiments, report the checkpoint, image/data set, device, parameters, and output path when known. Keep claims consistent with `readme.md` and `docs/agent-handoff.md`.
