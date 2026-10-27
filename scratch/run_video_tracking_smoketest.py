"""
Video tracking smoke test using synthetic video.
Creates a short synthetic video of moving colored rectangles (simulating fish),
then runs YOLOv8 + ByteTrack tracking and verifies:
- tracker.process_video runs without error
- output fields are present
- counting logic initializes (real detection-based counting requires actual fish video)

NOTE: This is explicitly a SMOKE TEST, not a validated tracking evaluation.
Actual counting validation requires annotated underwater video footage.
"""
import sys, os, cv2, json, tempfile, numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tracking.fish_tracker import FishVideoTracker
from src.preprocessing.enhancement import UnderwaterEnhancer
from src.utils.device import get_optimal_device
from ultralytics import YOLO

# --- Create minimal synthetic video: 640x480, 30 frames, blue background ---
tmpf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
tmpf.close()
video_path = tmpf.name

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

for i in range(30):
    frame = np.full((480, 640, 3), (50, 100, 80), dtype=np.uint8)
    # Draw 2 moving rectangles (no fish content, so detections expected = 0)
    x = int(50 + i * 10)
    cv2.rectangle(frame, (x, 100), (x + 60, 160), (0, 165, 255), -1)
    cv2.rectangle(frame, (500 - x // 2, 300), (560 - x // 2, 360), (255, 127, 0), -1)
    writer.write(frame)
writer.release()

print(f"Synthetic video created: {video_path} (30 frames, 640x480, no fish content)")

# --- Run tracker ---
device = get_optimal_device()
model = YOLO('models/best.pt')
enhancer = UnderwaterEnhancer()

tracker = FishVideoTracker(
    model=model,
    enhancer=enhancer,
    enhancement_method='raw',
    conf_threshold=0.25
)

output_path = video_path.replace('.mp4', '_tracked.mp4')
try:
    results = tracker.process_video(
        input_video_path=video_path,
        output_video_path=output_path,
        line_start=(0.05, 0.5),
        line_end=(0.95, 0.5),
        max_frames=30
    )
    print("\n--- Smoke Test Results ---")
    print(json.dumps({k: v for k, v in results.items() if k != 'crossing_events'}, indent=2))
    print(f"Output video written: {os.path.exists(output_path)}")

    # Schema checks
    assert 'total_frames_processed' in results
    assert 'total_fish_counted' in results
    assert 'species_counts' in results
    assert isinstance(results['crossing_events'], list)
    assert results['total_frames_processed'] == 30, f"Expected 30 frames, got {results['total_frames_processed']}"

    print("\n[PASS] FishVideoTracker smoke test passed.")
    print("[NOTE] Synthetic video has no fish. 0 counted is correct.")
    print("[LIMITATION] ByteTrack ID persistence and counting on real fish video not verified here.")
    print("             Ground-truth annotated video is required for MOTA/IDF1 tracking evaluation.")

except Exception as e:
    print(f"[FAIL] Tracking error: {e}")
    raise
finally:
    for p in [video_path, output_path]:
        try: os.remove(p)
        except: pass
