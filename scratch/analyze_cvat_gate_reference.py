"""Count ground-truth fish tracks crossing a horizontal gate in CVAT XML.

This uses only the Python standard library. It produces reference counts; it does
not run a detector or tracker and must not be presented as model performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def analyze_reference(
    annotation_path: Path,
    *,
    line_y_fraction: float = 0.5,
    gate_start_x_fraction: float = 0.05,
    gate_end_x_fraction: float = 0.95,
) -> dict:
    """Return per-track crossings using consecutive visible, unoccluded boxes."""
    root = ET.parse(annotation_path).getroot()
    width = int(root.findtext("./meta/original_size/width", "0"))
    height = int(root.findtext("./meta/original_size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError("CVAT annotations must include positive original image dimensions")
    if not 0 <= line_y_fraction <= 1:
        raise ValueError("line_y_fraction must be between 0 and 1")
    if not 0 <= gate_start_x_fraction < gate_end_x_fraction <= 1:
        raise ValueError("gate x fractions must satisfy 0 <= start < end <= 1")

    line_y = line_y_fraction * height
    gate_min_x = gate_start_x_fraction * width
    gate_max_x = gate_end_x_fraction * width
    crossing_events = []
    tracks = root.findall("./track")
    box_count = sum(len(track.findall("./box")) for track in tracks)

    for track in tracks:
        # CVAT marks fully outside or occluded frames explicitly. Excluding those
        # boxes avoids inventing a visible centroid while establishing references.
        points = []
        for box in track.findall("./box"):
            if box.get("outside") == "1" or box.get("occluded") == "1":
                continue
            x1, y1 = float(box.get("xtl")), float(box.get("ytl"))
            x2, y2 = float(box.get("xbr")), float(box.get("ybr"))
            points.append(
                {
                    "frame": int(box.get("frame")),
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                }
            )
        points.sort(key=lambda point: point["frame"])

        # The application counts a tracker ID once for the full video, regardless
        # of crossing direction, so the reference applies the same unique-track rule.
        for previous, current in zip(points, points[1:]):
            previous_side = previous["y"] - line_y
            current_side = current["y"] - line_y
            if not (
                (previous_side < 0 < current_side)
                or (current_side < 0 < previous_side)
            ):
                continue

            # Interpolate the centroid's x coordinate where its path meets the gate.
            fraction = (line_y - previous["y"]) / (current["y"] - previous["y"])
            crossing_x = previous["x"] + fraction * (current["x"] - previous["x"])
            if not gate_min_x <= crossing_x <= gate_max_x:
                continue
            crossing_events.append(
                {
                    "track_id": track.get("id"),
                    "frame_before": previous["frame"],
                    "frame_after": current["frame"],
                    "direction": "down" if current_side > previous_side else "up",
                    "crossing_x": round(crossing_x, 2),
                }
            )
            break

    metadata = root.find("./meta/job")
    start_frame = int(metadata.findtext("start_frame", "0")) if metadata is not None else 0
    stop_frame = int(metadata.findtext("stop_frame", "0")) if metadata is not None else 0
    return {
        "status": "reference_only_model_not_run",
        "annotation_file": str(annotation_path),
        "video_width": width,
        "video_height": height,
        "annotated_start_frame": start_frame,
        "annotated_stop_frame": stop_frame,
        "annotated_frame_count": stop_frame - start_frame + 1,
        "track_count": len(tracks),
        "box_count": box_count,
        "reference_filter": "outside=0 and occluded=0",
        "crossing_rule": "centroid crosses gate between consecutive retained boxes; one event per ground-truth track",
        "gate": {
            "line_y_fraction": line_y_fraction,
            "gate_start_x_fraction": gate_start_x_fraction,
            "gate_end_x_fraction": gate_end_x_fraction,
            "line_y_pixels": line_y,
            "start_x_pixels": gate_min_x,
            "end_x_pixels": gate_max_x,
        },
        "expected_total_crossings": len(crossing_events),
        "crossing_events": crossing_events,
        "limitations": [
            "This is ground-truth event extraction only; no detector or tracker was run.",
            "The single-class fish annotations do not support 13-species accuracy evaluation.",
            "A reference crossing across an annotation gap is based on the consecutive retained visible boxes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Path to CVAT video 1.1 annotations.xml")
    parser.add_argument("--output", type=Path, help="Optional JSON output path; refuses to overwrite")
    parser.add_argument("--line-y", type=float, default=0.5, help="Normalized horizontal gate height")
    parser.add_argument("--gate-start-x", type=float, default=0.05, help="Normalized gate start x")
    parser.add_argument("--gate-end-x", type=float, default=0.95, help="Normalized gate end x")
    args = parser.parse_args()

    result = analyze_reference(
        args.annotations,
        line_y_fraction=args.line_y,
        gate_start_x_fraction=args.gate_start_x,
        gate_end_x_fraction=args.gate_end_x,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        if args.output.exists():
            parser.error(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
