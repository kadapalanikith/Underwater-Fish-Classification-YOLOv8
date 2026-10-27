# An Evidence-Bounded Evaluation of YOLOv8 for Underwater Fish Detection

**Manuscript status:** project research draft; not peer reviewed or published.
**Evidence cut-off:** 2026-10-28.
**Project:** DeepSea AI — Underwater Fish Detection.

## Abstract

This project evaluates a saved YOLOv8n detector for 13 fish classes and records the limits of the available evidence. The local dataset contains 8,229 images. On a leakage-filtered subset of the local test split (399 images, 639 instances), the checkpoint achieved mAP@50 of 0.841 (image-bootstrap 95% CI 0.823–0.880) and mAP@50–95 of 0.636 (95% CI 0.615–0.694). The subset is not proven source-independent. A labeled preprocessing comparison found no supported improvement over a JPEG-95 control; gray-world and fusion preprocessing had supported mAP@50–95 degradations after Holm correction. On the external OBSEA test split, class-agnostic zero-shot fish detection achieved mAP@50 0.036 and mAP@50–95 0.017, indicating a substantial domain shift under this protocol. This is not species-level accuracy. The available RT-DETR result is timing-only because its checkpoint is COCO-pretrained, not fish-fine-tuned. On the annotated DIGI4ECO clip, the tested checkpoint produced zero track IDs and missed all six annotated gate crossings at both confidence 0.25 and 0.001. These findings support a carefully scoped local detector demonstration, not broad generalization, a fair detector comparison, or validated fish counting.

**Keywords:** underwater computer vision, fish detection, YOLOv8, domain shift, bootstrap uncertainty, image enhancement.

## 1. Introduction

Underwater images can have color distortion, reduced contrast, turbidity, variable illumination, and small or occluded targets. This project combines a saved YOLOv8 detector with a FastAPI service and React/Vite interface, and includes experiments for image enhancement, external-domain evaluation, latency, and video tracking. This manuscript distinguishes measured outputs from demonstrations and unexecuted plans so that the interface and experiment code are not mistaken for evidence of model performance.

The study asks:

1. What detection metrics are recorded for the saved checkpoint on the available local test data?
2. Do the tested image enhancements improve detection on the same evaluation images?
3. How does the detector behave on one external fish dataset when evaluated as a class-agnostic detector?
4. Which advertised comparisons and application features remain unvalidated?

## 2. Materials and methods

### 2.1 Local data and checkpoint

The project dataset is the 13-class fish detection dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/zehraatlgan/fish-detection/data). The repository records 8,229 images split into 6,834 train, 698 validation, and 697 test images. The detector checkpoint is `models/best.pt` (YOLOv8n; SHA-256 `af229161766094f3f002ff301b68001f6bec84492fe4f43c41c9f0838d1a5d23`).

The dataset contains both standard five-field YOLO box rows and polygon rows. For the saved evaluation artifacts, polygon annotations were converted to their enclosing axis-aligned boxes. This changes annotation geometry for 84 validation and 29 test instances. Evaluation therefore measures agreement with the converted boxes, not the original polygon geometry.

### 2.2 Leakage audit and held-out evaluation

The audit found 34 test images with exact train-image matches (33 distinct hashes), 212 test images sharing a source stem with training data (202 distinct prefixes), and 70 test images sharing a source stem with validation data (69 distinct prefixes). `test_clean` excludes 298 test images implicated by the filtering rules, leaving 399 images and 639 instances. This is a conservative leakage-filtered subset; source-level independence is not established.

The held-out evaluation used Ultralytics validation on CPU with `imgsz=640`, `conf=0.001`, `iou=0.6`, and `workers=0`. Uncertainty was estimated by 2,000 image-level bootstrap resamples with seed `20261027`; source-stem cluster intervals were reported as a sensitivity check. Overall mAP intervals only include resamples where every fixed evaluated class is present. This condition reduces the finite resample count and does not correct dataset leakage.

### 2.3 Image-enhancement comparison

The same checkpoint and 399 `test_clean` images were evaluated using original JPEGs, a JPEG-95 re-encoding control, CLAHE, gray-world color balance, and fusion enhancement. Enhanced images and the re-encoding control were saved as JPEG quality 95. Paired image-bootstrap comparisons used the JPEG-95 control, with Holm correction across three enhancement methods and two metrics. The supported-degradation rule required both an interval excluding zero and Holm-adjusted p < 0.05.

### 2.4 External-domain evaluation

The external evaluation used the test split of the [OBSEA fish detection dataset](https://doi.org/10.5281/zenodo.14888440), released under CC BY 4.0. The protocol retained all 618 test images, excluded annotations for Diver and Octopus vulgaris from fish ground truth, removed four duplicate fish boxes, and evaluated 5,369 fish instances. Predictions and labels were evaluated class-agnostically (`single_cls=True`, `agnostic_nms=True`) using `imgsz=640`, `conf=0.001`, `iou=0.6`, `max_det=300`, CPU, and no OBSEA tuning or model selection. Consequently, this result is not 13-class species accuracy. Predictions on excluded non-fish objects count as false positives under this setup.

### 2.5 Software and evidence boundaries

The project includes a FastAPI API, a React/Vite client, image enhancement code, a timing benchmark, and a video tracker. The saved RT-DETR checkpoint is COCO-pretrained and has not been fine-tuned on the fish data. The video tracking script's existing smoke test creates synthetic rectangles rather than fish footage. The current workspace has no Git metadata. These facts limit the claims made here.

## 3. Results

### 3.1 Local detection

| Split | Images / instances | Precision | Recall | mAP@50 (95% CI) | mAP@50–95 (95% CI) | Evidence status |
|---|---:|---:|---:|---:|---:|---|
| `valid_full` | 698 / 1,193 | 0.880 | 0.875 | 0.910 (0.886–0.939) | 0.658 (0.637–0.698) | Used during training/model selection; not independent. No AngelFish validation instances. |
| `test_full` | 697 / 1,129 | 0.816 | 0.804 | 0.840 (0.826–0.874) | 0.622 (0.604–0.670) | Contains identified train/validation overlap. |
| `test_clean` | 399 / 639 | 0.809 | 0.803 | 0.841 (0.823–0.880) | 0.636 (0.615–0.694) | Leakage-filtered; not proven source-independent. |

These are separate re-evaluation results and should not be merged with the historical notebook validation log (precision 0.865, recall 0.874, mAP@50 0.908, mAP@50–95 0.666). The historical log reports validation performance for 12 of 13 classes; AngelFish had no validation instances. The point estimates from the bootstrap artifacts reproduce the held-out metrics within maximum absolute delta 4.60 × 10⁻⁷.

### 3.2 Image enhancement

| Method | mAP@50 | mAP@50–95 | Mean enhancement time (ms/image) | Interpretation |
|---|---:|---:|---:|---|
| Original JPEG | 0.841 | 0.636 | — | Reference images |
| JPEG-95 control | 0.841 | 0.636 | — | Re-encoding control |
| CLAHE | 0.846 | 0.634 | 11.9 | No supported improvement; approximately neutral in this comparison |
| Gray-world | 0.812 | 0.613 | 10.5 | Supported mAP@50–95 degradation versus JPEG-95 after Holm correction |
| Fusion | 0.815 | 0.611 | 15.1 | Supported mAP@50–95 degradation versus JPEG-95 after Holm correction |

No tested enhancement showed a statistically supported improvement. The result is specific to one checkpoint, one filtered subset, and CPU inference; low-support classes limit class-specific conclusions.

### 3.3 External OBSEA evaluation

On 618 images and 5,369 fish instances, the class-agnostic zero-shot evaluation recorded precision 0.124, recall 0.094, mAP@50 0.036, mAP@50–95 0.017, and mAP@75 0.013, at a mean inference time of 29.7 ms/image on CPU. This is evidence of substantial domain shift for this checkpoint and protocol on one Mediterranean dataset. It does not demonstrate universal cross-domain behavior or species recognition on OBSEA.

### 3.4 Timing, tracking, and application checks

The timing-only CPU benchmark used 30 `test_clean` images, three warmup images, `imgsz=640`, confidence 0.25, and batch size 1. Mean inference times were 60.74 ms/image for the fish-trained YOLOv8 checkpoint and 706.32 ms/image for the COCO-pretrained RT-DETR checkpoint. Since RT-DETR was not fish-fine-tuned and the benchmark did not assess accuracy, it is not a model-accuracy comparison.

The test suite, frontend build, and route checks are software/code-path checks, not measurements of fish accuracy. On the DIGI4ECO OBSEA clip, CVAT provides 150 annotated frames, 12 tracks, and 1,384 box entries; the video decodes to 148 frames. Fourteen visible, non-occluded annotation boxes beyond the decodable video were excluded, leaving 1,327 boxes for comparison. The reference parser found six unique ground-truth crossings at the default center gate. The primary inference run (`conf=0.25`) produced zero detections and IDs; the sensitivity run (`conf=0.001`) produced 450 raw boxes, mean confidence 0.0041, but no IDs because every score was below ByteTrack's `track_low_thresh=0.1`. Both runs predicted zero crossings, for an absolute count error of six (100%) and event F1 of zero. Primary and sensitivity results are saved in `outputs/digi4eco_obsea_tracking_primary_v2.json` and `outputs/digi4eco_obsea_tracking_conf001_thresholds.json`. This is evidence that the current model/configuration failed on this clip, not evidence of persistent-ID performance.

## 4. Discussion

The local evaluation records useful performance for the saved checkpoint on this dataset, but overlap and provenance constraints prevent a strong independent-test claim. The leakage-filtered subset reduces known overlap and supports uncertainty estimation, yet it cannot establish source independence. Differences between full and filtered results should not be interpreted as evidence that leakage has no effect.

The enhancement experiment does not support a general claim that preprocessing improves this detector. CLAHE was approximately neutral, while gray-world and fusion degraded mAP@50–95 under the paired corrected analysis. The OBSEA result also warns that a locally strong score does not guarantee transfer to different cameras, environments, and taxonomies.

The RT-DETR timing numbers compare inference runtimes for checkpoints trained for different label spaces. They cannot rank fish detection quality. The tracker run is a real-video negative result: the tested detector produced no track IDs and missed all annotated crossings. The synthetic tests demonstrate code paths only and do not validate natural-video tracking accuracy.

## 5. Limitations and remaining validation

1. The local `test_clean` set is not proven source-independent; obtain an independently collected, labeled local test set for a stronger estimate.
2. OBSEA is one external dataset evaluated class-agnostically. A second independently labeled domain and taxonomy-aware protocol are needed before making broader transfer claims.
3. A fair RT-DETR accuracy study requires fish fine-tuning, a verified compatible GPU environment, and a matched evaluation protocol. No such training has been run.
4. The DIGI4ECO clip is from the same OBSEA domain as the external image evaluation. The video decodes to 148 frames although annotations cover 150; 14 boxes beyond the clip were excluded. The tested model produced no track IDs and missed all six crossings. Threshold or tracker tuning must use separate development footage before a new held-out video is used for evaluation.
5. The environment lacks a Python lock file; exact dependency recreation is not guaranteed.
6. No Git metadata is present in this workspace, preventing a tracked-history or Git-history secret audit.
7. Existing metrics are saved artifacts described by their provenance. They have not all been rerun in this manuscript preparation pass.

## 6. Conclusion

The project provides a fish-detection application and documented local, enhancement, external-dataset, and real-video evaluations. The recorded local `test_clean` mAP@50 is 0.841, with bootstrap uncertainty and known leakage limitations. Tested enhancement methods did not provide a supported accuracy improvement. Zero-shot class-agnostic OBSEA performance was low and should be presented as one domain-shift result. RT-DETR accuracy remains untested, and the current real-video tracker failed to produce IDs or crossings. The appropriate conclusion is a research prototype with useful, bounded evidence—not a completed general-purpose fish detector or validated counting system.

## Reproducibility and artifact index

| Evidence | Script or artifact |
|---|---|
| Dataset audit and overlap | `scratch/run_dataset_audit.py`, `outputs/dataset_audit.json` |
| Held-out YOLO evaluation | `scratch/run_heldout_evaluation.py`, `outputs/heldout_yolov8_metrics.json` |
| Bootstrap uncertainty | `scratch/run_bootstrap_uncertainty.py`, `outputs/bootstrap_uncertainty.json`, `outputs/uncertainty_stats/` |
| Enhancement accuracy | `scratch/run_enhancement_accuracy.py`, `outputs/enhancement_accuracy_test_clean.json` |
| OBSEA evaluation | `scratch/run_obsea_evaluation.py`, `outputs/obsea_evaluation.json` |
| Timing-only benchmark | `scratch/run_latency_benchmark.py`, `outputs/latency_yolov8_vs_rtdetr.json` |
| Current project status and commands | `readme.md`, `docs/agent-handoff.md` |

The command recorded for the Python suite is `python -m pytest tests -v`. The project handoff records 57 passing tests in 11.30 seconds in its `.venv`; that run was reported on 2026-10-28 and is not a new execution performed while drafting this manuscript. The Codex audit shell could not reproduce it: its `.venv` points to a missing Python 3.11 executable, while system Python 3.14.7 stopped collection with two `cv2` import errors (14 items collected; no tests ran). The handoff also records a passing Vite build and route smoke check. Re-run these commands in the target clone before release.

## References

1. Zehra Atlgan, [Fish Detection dataset](https://www.kaggle.com/datasets/zehraatlgan/fish-detection/data), Kaggle. Dataset used by this project; see the dataset page for its source and terms.
2. Bañós Castelló, P. et al., [Underwater images from OBSEA fish detection training dataset (YOLO)](https://doi.org/10.5281/zenodo.14888440), Zenodo (2025). DOI: 10.5281/zenodo.14888440. CC BY 4.0.
3. Dawkins, M. et al., [FishTrack23: An Ensemble Underwater Dataset for Multi-Object Tracking](https://openaccess.thecvf.com/content/WACV2024/html/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.html), WACV 2024. Cited as relevant tracking-dataset context; FishTrack23 was not used in the experiments reported here.
4. González-Villà, S. et al., [DIGI4ECO Underwater Fish Tracking Dataset](https://doi.org/10.5281/zenodo.20701225), Zenodo (2026). DOI: 10.5281/zenodo.20701225. CC BY 4.0. The OBSEA video and annotations were used for a descriptive tracker evaluation; the tested checkpoint produced zero IDs/counts under the two documented confidence settings.

## Citation note

This Markdown is a project research draft, not a peer-reviewed paper. Cite the underlying datasets and methods when reusing them. Do not present this manuscript as accepted, published, or peer reviewed.
