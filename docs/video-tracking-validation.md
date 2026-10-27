# Video Tracking Validation Protocol

Status: **real-video processing completed; tracker failed to detect/track the annotated fish under tested settings**. The user approved the DIGI4ECO archive download. The local archive matches the Zenodo API MD5. Only `OBSEA/video.mp4` and `OBSEA/annotations.xml` were extracted. CVAT has 12 tracks, 1,384 boxes, and labels through frame 149, while the video decodes to 148 frames (0–147). Fourteen visible, non-occluded boxes beyond the video were excluded, leaving 1,327 boxes for frame-level comparison. This clip is from the OBSEA domain already represented by the project's external image evaluation; it is not a second independent domain.

The run used isolated `.venv-video` with Python 3.11.16, PyTorch 2.12.0+cpu, Ultralytics 8.4.190, and OpenCV 5.0.0. CUDA was unavailable by design. See the observed outcomes below; this is a negative validation result, not successful fish tracking.

## Available ground-truth reference

The CVAT metadata reports 150 annotated frames at 1912×1080. With the default gate (`line_y=0.5`, `line_start=(0.05, 0.5)`, `line_end=(0.95, 0.5)`), the parser found **6 unique ground-truth track crossings** from visible, non-occluded boxes. Event pairs occur at frames 86–87, 12–13, 31–32, 45–46, 80–81, and 72–73. The saved reference is `outputs/digi4eco_obsea_gate_reference.json`.

## Executed results

| Run | Confidence | Raw boxes | Assigned IDs / tracked boxes | Predicted crossings | Count absolute error | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Primary, ByteTrack defaults | 0.25 | 0 | 0 / 0 | 0 | 6 (100%) | 10.206 s; 68.961 ms/frame |
| Sensitivity, ByteTrack defaults | 0.001 | 450 | 0 / 0 | 0 | 6 (100%) | 10.1 s; 68.242 ms/frame |

At confidence 0.001, mean raw confidence was 0.0041 (min 0.001, max 0.0771), below ByteTrack's `track_low_thresh=0.1`, explaining why none became track IDs. Both results have event precision/recall/F1 of 0. The primary JSON is `outputs/digi4eco_obsea_tracking_primary_v2.json`; sensitivity JSON is `outputs/digi4eco_obsea_tracking_conf001_thresholds.json`. These settings were evaluated on this same clip; do not tune on it and then present it as an untouched test set. The result does not establish persistent-ID accuracy because the detector produced no tracks.

Reproduce the reference in a Python environment with no third-party packages:

```powershell
python scratch\analyze_cvat_gate_reference.py external_data\digi4eco\extracted\OBSEA\annotations.xml --output outputs\digi4eco_obsea_gate_reference.json
```

The result is `outputs/digi4eco_obsea_gate_reference.json`. The 13-class project model has no fish-species ground truth for this single-class sequence, so the real-video comparison must report total crossing counts only. Do not claim species accuracy from this clip.

## Required inputs

1. The acquired underwater video file `external_data/digi4eco/extracted/OBSEA/video.mp4` and matching CVAT ground-truth file `external_data/digi4eco/extracted/OBSEA/annotations.xml`.
2. A reference file for the same video, containing either:
   - crossing counts by species and total, or
   - a list of reference crossing events with frame numbers, or
   - frame-level boxes and track IDs for MOTA/IDF1/HOTA.
3. The exact counting line used for the reference annotation.

Reference JSON template for count/event validation:

```json
{
  "video": "relative/or/absolute/path/to/video.mp4",
  "line_start": [0.05, 0.5],
  "line_end": [0.95, 0.5],
  "expected_total": 2,
  "expected_species_counts": {
    "ClownFish": 1,
    "BlueTang": 1
  },
  "crossings": [
    {"frame": 31, "species": "ClownFish"},
    {"frame": 87, "species": "BlueTang"}
  ]
}
```

`crossings` is optional. When present, it must be consistent with `expected_total` and `expected_species_counts`; the validator must report any mismatch rather than silently choosing one source.

For MOTA/IDF1/HOTA, use MOTChallenge-style per-frame boxes and track IDs. Those metrics require a tracking evaluator such as TrackEval; it is not installed in this workspace, so no tracking-accuracy metric has been computed.

## Protocol

1. Record the video SHA-256, frame count, resolution, and FPS.
2. Record the checkpoint path and SHA-256. The current fish checkpoint is `models/best.pt`, SHA-256 `af229161766094f3f002ff301b68001f6bec84492fe4f43c41c9f0838d1a5d23`.
3. Use the reference line coordinates exactly. The current API exposes only a horizontal line via `line_y`; `FishVideoTracker.process_video` accepts arbitrary normalized `line_start` and `line_end`.
4. Run the primary configuration: `enhancement=raw`, `conf=0.25`, ByteTrack, the same `max_frames` as the reference, and CPU unless a compatible GPU environment is separately verified.
5. Save the full result dictionary, including `crossing_events`, and the annotated output video when requested.
6. Repeat only as a sensitivity check with `clahe`; do not select the enhancement method on the evaluation video.
7. Keep the reference file and result JSON together. Do not report a metric without both.

Example using the tracker directly:

```powershell
.\.venv\Scripts\python.exe -c "import json; from ultralytics import YOLO; from src.preprocessing.enhancement import UnderwaterEnhancer; from src.tracking.fish_tracker import FishVideoTracker; from src.utils.device import get_optimal_device; m=YOLO('models/best.pt'); t=FishVideoTracker(m, UnderwaterEnhancer(), enhancement_method='raw', conf_threshold=0.25, device=get_optimal_device()); r=t.process_video('path/to/video.mp4', 'outputs/video_tracked.mp4', line_start=(0.05, 0.5), line_end=(0.95, 0.5), max_frames=1000); open('outputs/video_tracking_result.json','w',encoding='utf-8').write(json.dumps(r, indent=2))"
```

The API path is `POST /api/track` with form fields `file`, `enhancement`, `model_name`, `conf`, `line_y`, and `max_frames`. It currently supports only a horizontal line and defaults to `max_frames=100`; set it explicitly for a full-video run.

## Minimum metrics

- Frames processed versus reference frames.
- Total predicted crossings versus `expected_total`: signed error, absolute error, and relative absolute error when the expected total is nonzero.
- Per-species predicted counts versus reference counts, including omitted and extra species.
- Event precision, recall, and F1 after one-to-one matching within a declared frame tolerance. Report the tolerance; do not match across the whole video.
- Failure modes: dropped frames, ID switches, duplicate crossings, missed crossings, and false crossings. These are qualitative unless frame-level reference tracks are available.
- Runtime, device, and enhancement method.

No acceptance threshold is set here. Pre-register the threshold with the user before looking at results; otherwise report descriptive error only and do not claim tracking accuracy.

## Artifacts to save

- `outputs/video_tracking_validation.json`: video/checkpoint hashes, reference hash, settings, result dictionary, metrics, limitations.
- `outputs/video_tracked.mp4`: annotated output, if requested.
- The original video and reference JSON, unchanged.

## Current blocker

The real clip was processed, but fish tracking/counting failed under both tested confidence settings. The synthetic smoke test remains a code-path check only. To study ID persistence, first use separate development footage to select detector/tracker thresholds, then evaluate on a new held-out video with frame-level IDs. Do not claim successful real-video tracking from the current zero-track result.
