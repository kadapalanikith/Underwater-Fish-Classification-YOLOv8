"""Focused tests for video tracking IDs, counting, and input/error handling."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.tracking.fish_tracker import FishVideoTracker


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def int(self):
        return FakeTensor(self.values.astype(int))


class FakeBoxes:
    def __init__(self, xyxy, track_id):
        self.xyxy = FakeTensor([xyxy])
        self.id = FakeTensor([track_id])
        self.cls = FakeTensor([0])
        self.conf = FakeTensor([0.9])

    def __bool__(self):
        return True


class MovingTrackModel:
    """Return one stable ID whose center crosses the test video's gate."""

    def __init__(self):
        self.frame = 0
        self.centers_y = [100, 140, 180]

    def track(self, frame, **kwargs):
        center_y = self.centers_y[min(self.frame, len(self.centers_y) - 1)]
        self.frame += 1
        boxes = FakeBoxes([80, center_y - 10, 120, center_y + 10], track_id=7)
        return [SimpleNamespace(boxes=boxes, names={0: "Fish"})]


def make_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    try:
        for _ in range(3):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_persistent_id_counts_one_crossing_once(tmp_path):
    video = make_video(tmp_path / "three_frames.mp4")
    tracker = FishVideoTracker(MovingTrackModel(), device="cpu")

    result = tracker.process_video(
        str(video),
        line_start=(0.05, 0.5),
        line_end=(0.95, 0.5),
        max_frames=3,
        include_track_observations=True,
    )

    assert result["total_frames_processed"] == 3
    assert result["active_tracks_tracked"] == 1
    assert result["total_fish_counted"] == 1
    assert result["crossing_events"][0]["track_id"] == 7
    assert [frame["tracks"][0]["track_id"] for frame in result["track_observations"]] == [7, 7, 7]


def test_rejects_missing_video_before_inference():
    tracker = FishVideoTracker(model=None, device="cpu")

    with pytest.raises(ValueError, match="Could not open video"):
        tracker.process_video("missing_underwater_video.mp4")


def test_rejects_zero_frame_limit():
    tracker = FishVideoTracker(model=None, device="cpu")

    with pytest.raises(ValueError, match="max_frames must be a positive integer"):
        tracker.process_video("unused.mp4", max_frames=0)
