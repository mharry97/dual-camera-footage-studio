"""
Export camera extrinsics from a left/right video pair.

Runs the same feature-matching calibration pipeline used for stitching, then
extracts the raw rotation matrices and intrinsics and writes them to a JSON file.

Output JSON contains:
  - R matrix (3x3) for each camera — orientation in the shared world frame
  - K matrix (3x3) for each camera — intrinsics as estimated by bundle adjustment
  - fov — horizontal/vertical field of view in radians (derived from K + frame dims)
  - confidence — match confidence from the feature matcher

Note: distortion coefficients are not produced here — they require a separate
checkerboard calibration (cv2.calibrateCamera).

Usage:
    uv run export-extrinsics LEFT.mp4 RIGHT.mp4 OUTPUT.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from stitching.warper import Warper as OsWarper

from footage_studio.stitching.calibration import (
    MEDIUM_MEGAPIX,
    MAX_ATTEMPTS,
    WINDOW_DURATION,
    _CalibrationStitcher,
    _window_starts,
)
from footage_studio.stitching.video_io import probe_video, read_frame_at

_DETECTOR_CONFIGS = [
    {"detector": "orb",   "nfeatures": 1000, "confidence_threshold": 0.1},
    {"detector": "orb",   "nfeatures": 3000, "confidence_threshold": 0.1},
    {"detector": "sift",  "nfeatures": 1000, "confidence_threshold": 0.1},
    {"detector": "akaze", "nfeatures": 1000, "confidence_threshold": 0.1},
    {"detector": "akaze", "nfeatures": 1000, "confidence_threshold": 0.01},
    {"detector": "sift",  "nfeatures": 3000, "confidence_threshold": 0.01},
]


def _load_npz_K(npz_path: Path, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Load K and distCoeff from an npz, scaling cy for the actual frame height."""
    npz = np.load(str(npz_path))
    K   = npz["intrinsic_matrix"].astype(np.float64).copy()
    dist = npz["distCoeff"].reshape(1, 14).astype(np.float64)
    _NPZ_CAL_H = 2160
    if frame_h != _NPZ_CAL_H:
        K[1, 2] += (frame_h - _NPZ_CAL_H) / 2.0
    return K, dist


def export_extrinsics(left: Path, right: Path, npz_path: Path | None = None) -> dict:
    """
    Run calibration on a left/right video pair and return raw camera parameters.

    Returns a dict suitable for json.dump with R matrices, K matrices, FoV bounds,
    and confidence.
    """
    frame_w, frame_h, _, duration = probe_video(left)

    full_megapix = (frame_w * frame_h) / 1e6
    medium_scale = min(1.0, math.sqrt(MEDIUM_MEGAPIX / full_megapix))
    aspect = 1.0 / medium_scale

    last_error: Exception | None = None
    stitcher = None

    npz_K, npz_dist = _load_npz_K(npz_path, frame_h) if npz_path is not None else (None, None)

    for window_start in _window_starts(duration):
        sample_time = window_start + WINDOW_DURATION / 2
        frame_left = read_frame_at(left, sample_time, frame_w, frame_h)
        frame_right = read_frame_at(right, sample_time, frame_w, frame_h)
        if frame_left is None or frame_right is None:
            continue

        if npz_K is not None:
            frame_left  = cv2.undistort(frame_left,  npz_K, npz_dist, None, npz_K)
            frame_right = cv2.undistort(frame_right, npz_K, npz_dist, None, npz_K)

        for cfg in _DETECTOR_CONFIGS:
            try:
                s = _CalibrationStitcher(**cfg)
                s.stitch([frame_left, frame_right])
                stitcher = s
                break
            except Exception as e:
                last_error = e

        if stitcher is not None:
            break

    if stitcher is None:
        raise RuntimeError(f"Calibration failed: {last_error}")

    cameras = stitcher._cameras
    confidence = stitcher._confidence

    results = []
    for camera in cameras:
        K = npz_K if npz_K is not None else OsWarper.get_K(camera, aspect)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # Field of view in radians, accounting for principal point offset
        fov_left   =  math.atan(cx / fx)
        fov_right  =  math.atan((frame_w - cx) / fx)
        fov_top    =  math.atan(cy / fy)
        fov_bottom =  math.atan((frame_h - cy) / fy)

        results.append({
            "R": camera.R.tolist(),
            "K": K.tolist(),
            "fov": {
                "left":   -fov_left,
                "right":   fov_right,
                "top":     fov_top,
                "bottom": -fov_bottom,
            },
        })

    return {
        "frame_w": frame_w,
        "frame_h": frame_h,
        "confidence": confidence,
        "left": results[0],
        "right": results[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export camera extrinsics for veo-reverse-engineered")
    parser.add_argument("left", type=Path, help="Left lens video")
    parser.add_argument("right", type=Path, help="Right lens video")
    parser.add_argument("output", type=Path, help="Output JSON file")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Camera calibration npz — undistorts frames before matching for consistent R matrices")
    args = parser.parse_args()

    print(f"Running calibration on {args.left.name} / {args.right.name}...")
    data = export_extrinsics(args.left, args.right, args.npz)
    print(f"Confidence: {data['confidence']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Written → {args.output}")


if __name__ == "__main__":
    main()
