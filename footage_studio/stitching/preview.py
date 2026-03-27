"""
Quick stitch preview: calibrate and render a single frame to a JPEG.

Outputs to {output_dir}/preview/{name}_preview.jpg alongside _left.jpg and _right.jpg.

Usage:
    uv run python -m footage_studio.stitching.preview LEFT RIGHT [--time SECONDS]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from footage_studio.core import get_output_dir
from footage_studio.stitching.calibration import calibrate
from footage_studio.stitching.pipeline import _apply_gain, _precompute_blend_weights
from footage_studio.stitching.video_io import probe_video, read_frame_at


def preview_frame(left: Path, right: Path, time_s: float | None = None) -> Path:
    """
    Calibrate and render a single blended frame plus the raw left/right frames.
    Saves to {output_dir}/preview/ and returns the preview directory path.
    """
    output_dir = get_output_dir()
    if output_dir is None:
        print("Error: output directory not configured in settings.", file=sys.stderr)
        sys.exit(1)

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    name = left.stem

    print(f"Calibrating from {left.name} / {right.name}...")
    cal = calibrate(left, right)
    print(f"Calibration done — confidence: {cal.confidence:.3f}, canvas: {cal.out_w}x{cal.out_h}")

    frame_w, frame_h, _, duration = probe_video(left)
    if time_s is None:
        time_s = duration / 2

    print(f"Reading frame at {time_s:.1f}s...")
    frame_left = read_frame_at(left, time_s, frame_w, frame_h)
    frame_right = read_frame_at(right, time_s, frame_w, frame_h)
    if frame_left is None or frame_right is None:
        print("Failed to read frames at that time.", file=sys.stderr)
        sys.exit(1)

    out_w, out_h = cal.out_w, cal.out_h
    (map_x_0, map_y_0), (cx0, cy0), (ww0, wh0) = cal.maps[0], cal.corners[0], cal.warped_sizes[0]
    (map_x_1, map_y_1), (cx1, cy1), (ww1, wh1) = cal.maps[1], cal.corners[1], cal.warped_sizes[1]

    gain_0 = cal.gains[0] if cal.gains else np.ones(3, dtype=np.float32)
    gain_1 = cal.gains[1] if cal.gains else np.ones(3, dtype=np.float32)
    frame_left = _apply_gain(frame_left, gain_0)
    frame_right = _apply_gain(frame_right, gain_1)

    w0, w1 = _precompute_blend_weights(cal, frame_w, frame_h)
    w0_3ch = np.stack([w0, w0, w0], axis=-1)
    w1_3ch = np.stack([w1, w1, w1], axis=-1)

    warped_0 = cv2.remap(frame_left, map_x_0, map_y_0, cv2.INTER_LINEAR)
    warped_1 = cv2.remap(frame_right, map_x_1, map_y_1, cv2.INTER_LINEAR)

    canvas_0 = np.zeros((out_h, out_w, 3), dtype=np.float32)
    canvas_1 = np.zeros((out_h, out_w, 3), dtype=np.float32)
    canvas_0[cy0:cy0 + wh0, cx0:cx0 + ww0] = warped_0[:wh0, :ww0].astype(np.float32)
    canvas_1[cy1:cy1 + wh1, cx1:cx1 + ww1] = warped_1[:wh1, :ww1].astype(np.float32)

    result = (canvas_0 * w0_3ch + canvas_1 * w1_3ch).astype(np.uint8)

    cv2.imwrite(str(preview_dir / f"{name}_preview.jpg"), result)
    cv2.imwrite(str(preview_dir / f"{name}_left.jpg"), frame_left)
    cv2.imwrite(str(preview_dir / f"{name}_right.jpg"), frame_right)

    print(f"Saved to {preview_dir}/")
    print(f"  {name}_preview.jpg")
    print(f"  {name}_left.jpg")
    print(f"  {name}_right.jpg")

    return preview_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--time", type=float, default=None, help="Time in seconds (default: middle of video)")
    args = parser.parse_args()

    preview_frame(args.left, args.right, args.time)


if __name__ == "__main__":
    main()
