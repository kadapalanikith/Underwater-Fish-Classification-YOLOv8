# RT-DETR Fine-Tuning Plan

Status: planned, not executed. No package installation, weight download, GPU training, or accuracy claim has been made.

## Current evidence

- `models/rtdetr-l.pt` is a COCO-pretrained RT-DETR-L checkpoint: 32,970,476 parameters, 66,511,432 bytes, 80 COCO classes, SHA-256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`.
- `outputs/latency_yolov8_vs_rtdetr.json` is a timing-only comparison. It is not a fish-species accuracy comparison.
- The local fish checkpoint is `models/best.pt`, trained from the local 13-class dataset. Do not use its test results as RT-DETR evidence.
- Current `.venv` has `torch 2.14.1+cu126`, which lacks kernels for the RTX 5050 Laptop GPU (`sm_120`). The CPU fallback is not suitable for full RT-DETR training.

## Fair-comparison tiers

### Tier 1: preferred matched retrain

Train both YOLOv8n and RT-DETR-L from COCO-pretrained weights with the same local training split, 640 px input, 10 epochs, seed, augmentation policy, and effective batch size. Evaluate both on the same `test_clean` images (primary) and `test_full` (secondary) with the same confidence, IoU, and max-detection settings.

This requires a new `yolov8n.pt` COCO checkpoint download. The repository does not currently contain one. Do not download it without approval.

### Tier 2: budget-matched RT-DETR only

Fine-tune RT-DETR-L for 10 epochs on the local training split and compare with the saved YOLOv8 fish checkpoint. Label this as budget-matched but not identical: the original YOLOv8 training run, augmentation details, and hardware are not fully recorded. This is weaker evidence than Tier 1.

Primary decision: prefer Tier 1 if the user approves the small YOLOv8n weight download and the extra training time. Otherwise run Tier 2 and state the limitation explicitly.

## Environment gate

Do not modify the working `.venv`. Create a separate environment, for example `.venv-rtdetr`, from the existing Python 3.11 executable.

The current torch warning suggests CUDA 13.0, 13.2, or 13.4 wheels for this GPU. Verify wheel availability for Python 3.11 and driver 610.74 before installing. The exact install command must be checked against the package index; do not assume the wheel exists.

Required checks before any training:

```powershell
.\.venv-rtdetr\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv-rtdetr\Scripts\python.exe -c "import torch; x=torch.ones((512,512),device='cuda'); print((x@x).sum().item())"
nvidia-smi
```

A successful import or `torch.cuda.is_available()` is not enough. The tensor operation must succeed. If it fails, stop and record the error; do not silently fall back to CPU for a long run.

## Data protocol

- Training: 6,834 local training images.
- Validation during training: 698 local validation images.
- Final evaluation: `test_clean` (399 leakage-filtered images / 639 instances), then `test_full` (697 / 1,129).
- Keep the test sets out of training and model selection.
- Polygons are converted to enclosing boxes by `scratch/eval_common.py`.
- Use hard-linked derived copies under `outputs/derived/`; do not modify `Dataset/`.
- Use the same 13-class order and names for both models.

Prepare the RT-DETR training data without training:

```powershell
.\.venv\Scripts\python.exe scratch\run_rtdetr_finetune.py --prepare-data
```

## Training protocol

Start with a one-epoch pilot, not a full run:

```powershell
.\.venv-rtdetr\Scripts\python.exe scratch\run_rtdetr_finetune.py --pilot --execute --batch 2 --imgsz 640 --device 0
```

The pilot must record:

- Whether the GPU tensor gate passed.
- Peak allocated VRAM.
- Wall-clock time for one epoch.
- Whether AMP was active.
- Any out-of-memory or kernel errors.
- The produced checkpoint and run directory.

Suggested escalation only after the pilot:

1. Batch 2 at 640 with AMP.
2. If VRAM is below about 7 GiB, try batch 4.
3. If OOM, try batch 1 or 512 px; do not change evaluation resolution without reporting it.
4. If batch 1 still OOM, stop and report rather than freezing layers silently.

Full 10-epoch run, only after explicit approval:

```powershell
.\.venv-rtdetr\Scripts\python.exe scratch\run_rtdetr_finetune.py --train --execute --epochs 10 --batch 2 --imgsz 640 --device 0
```

Do not claim a completed comparison until both models have been evaluated on the same images and the reports are saved.

## Evaluation protocol

- Resize to 640 px.
- Ultralytics validation settings: `conf=0.001`, `iou=0.6`, `max_det=300`, `multi_label=True`.
- Report mAP@50, mAP@50-95, precision, recall, per-class AP, parameter count, checkpoint size, and inference latency.
- Use the same class mapping and image sets for YOLOv8 and RT-DETR.
- Keep `test_full` and `test_clean` separate; do not merge metrics.
- If the RT-DETR run is trained on `train` and validates on `valid`, selecting the best checkpoint on `valid` is allowed, but the final `test_clean` report must state that selection occurred on validation.
- Add image-level bootstrap intervals only after reviewing point estimates and runtime; reuse `src/evaluation/uncertainty.py` where the metric row format matches. Do not fabricate intervals for an evaluation that was not run.

Command for an already trained checkpoint:

```powershell
.\.venv-rtdetr\Scripts\python.exe scratch\run_rtdetr_finetune.py --evaluate outputs\rtdetr_runs\<run>\weights\best.pt --execute
```

Expected output: `outputs/rtdetr_finetune_metrics.json`, with `_metadata`, training settings, split results, and limitations.

## Storage estimate

- Separate Python environment: approximately 5-8 GB.
- CUDA/PyTorch wheel download: approximately 2.5-4 GB.
- Derived hard-linked data: negligible additional bytes.
- Checkpoints: about 66 MB each.
- Logs and plots: reserve about 1-2 GB.
- Practical reserve: 15 GB free. D: currently has about 622 GB free; C: about 18.6 GB free.

Use D: for the environment and outputs if C: becomes tight.

## Time estimate

Unknown until the pilot. Do not extrapolate from COCO benchmarks or from CPU latency. If one epoch takes `T` minutes, a 10-epoch run is approximately `10 * T` plus validation and checkpoint overhead. Record the measured pilot value and update this section.

## Risks and stop conditions

- No `sm_120` kernels in the current environment: stop before training.
- GPU OOM at batch 1 / 512 px: stop and report configuration options.
- No approved `yolov8n.pt`: do not run Tier 1; use Tier 2 only with its limitation.
- Data leakage or class mismatch: stop and inspect before training.
- Missing `test_clean` or `test_full` labels: do not substitute `qualitative_samples`.
- Any request to report RT-DETR as fish-accurate without fine-tuning: refuse and explain that the existing checkpoint is COCO-pretrained.

## Approval gates

1. Approve creation of `.venv-rtdetr` and installation of a CUDA 13.x torch wheel.
2. Approve the GPU tensor gate and one-epoch pilot.
3. Review pilot VRAM/time before approving the full run.
4. Approve a `yolov8n.pt` download if Tier 1 is required.
5. Review final metrics before changing application text or research claims.

No step above has been executed yet beyond checkpoint/environment inspection.
