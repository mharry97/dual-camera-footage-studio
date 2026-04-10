"""
Calibration using OpenStitching's camera estimation pipeline.

Runs feature detection, homography estimation, bundle adjustment, and wave
correction on a sample frame pair to produce per-camera rotation matrices and
intrinsics. These are used to build precomputed remap tables (via
cv2.PyRotationWarper.buildMaps) that can be applied to every video frame with
cv2.remap — no per-frame feature matching required.

Also computes seam masks (graph-cut optimal seam line) and exposure gain
corrections during calibration so they can be applied to every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
from stitching import Stitcher
from stitching.images import Images
from stitching.seam_finder import SeamFinder
from stitching.warper import Warper as OsWarper

from footage_studio.stitching.video_io import probe_video, read_frame_at

MAX_ATTEMPTS = 5
WINDOW_DURATION = 30.0
MEDIUM_MEGAPIX = Images.Resolution.MEDIUM.value  # 0.6
CONFIDENCE_THRESHOLD = 0.3
SEAM_MEGAPIX = 0.1  # resolution for seam finding (speed vs quality)


class _CalibrationStitcher(Stitcher):
    """
    Stops after camera estimation + wave correction + scale 
    """

    def stitch(self, images, feature_masks=[]):  # noqa: B006
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )
        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs, feature_masks)
        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        self.estimate_scale(cameras)
        self._cameras = cameras
        self._confidence = max(
            (m.confidence for m in matches),
            default=0.0,
        )


@dataclass
class CalibrationResult:
    """
    Precomputed data for stitching a left/right video pair.

    Index 0 = first input video (left), index 1 = second (right).

    maps:         remap tables for cv2.remap, one (map_x, map_y) per camera
    corners:      top-left position of each warped frame on the output canvas
    warped_sizes: (w, h) of each warped output
    seam_masks:   canvas-sized uint8 arrays (255 = use this camera, 0 = use other)
    gains:        per-channel float32 gains (shape (3,)) to apply before remapping
    """

    maps: list[tuple[np.ndarray, np.ndarray]]
    corners: list[tuple[int, int]]
    warped_sizes: list[tuple[int, int]]
    out_w: int
    out_h: int
    confidence: float
    seam_masks: list[np.ndarray] = field(default_factory=list)
    gains: list[np.ndarray] = field(default_factory=list)


def _window_starts(duration: float) -> list[float]:
    """Return MAX_ATTEMPTS evenly-distributed window start times, middle first."""
    section = duration / MAX_ATTEMPTS
    centers = [section * (i + 0.5) for i in range(MAX_ATTEMPTS)]
    starts = [max(0.0, c - WINDOW_DURATION / 2) for c in centers]
    mid = MAX_ATTEMPTS // 2
    order = [mid] + [i for i in range(MAX_ATTEMPTS) if i != mid]
    return [starts[i] for i in order]


def _build_remap_result(
    cameras: list,
    scale: float,
    warper_type: str,
    frame_w: int,
    frame_h: int,
    confidence: float,
    medium_scale: float,
) -> CalibrationResult:
    """Build CalibrationResult from camera params, without seam/exposure (added later)."""
    aspect = 1.0 / medium_scale
    warper = cv.PyRotationWarper(warper_type, scale * aspect)
    src_size = (frame_w, frame_h)

    maps = []
    corners_raw = []
    warped_sizes = []

    for camera in cameras:
        K = OsWarper.get_K(camera, aspect)
        corner, map_x, map_y = warper.buildMaps(src_size, K, camera.R)
        roi = warper.warpRoi(src_size, K, camera.R)
        maps.append((map_x, map_y))
        corners_raw.append((int(corner[0]), int(corner[1])))
        warped_sizes.append((int(roi[2]), int(roi[3])))

    min_x = min(c[0] for c in corners_raw)
    min_y = min(c[1] for c in corners_raw)
    max_x = max(c[0] + s[0] for c, s in zip(corners_raw, warped_sizes))
    max_y = max(c[1] + s[1] for c, s in zip(corners_raw, warped_sizes))

    out_w = int(max_x - min_x)
    out_h = int(max_y - min_y)
    out_w += out_w % 2
    out_h += out_h % 2

    corners = [(c[0] - min_x, c[1] - min_y) for c in corners_raw]

    return CalibrationResult(
        maps=maps,
        corners=corners,
        warped_sizes=warped_sizes,
        out_w=out_w,
        out_h=out_h,
        confidence=confidence,
    )


def _warp_frames(
    frames: list[np.ndarray],
    cal: CalibrationResult,
    frame_w: int,
    frame_h: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Warp calibration frames through remap tables. Returns (warped_images, validity_masks)."""
    solid = np.ones((frame_h, frame_w), dtype=np.uint8) * 255
    warped_imgs = []
    validity_masks = []
    for frame, (map_x, map_y), (ww, wh) in zip(frames, cal.maps, cal.warped_sizes):
        warped = cv.remap(frame, map_x, map_y, cv.INTER_LINEAR)
        mask = cv.remap(solid, map_x, map_y, cv.INTER_NEAREST)
        warped_imgs.append(warped[:wh, :ww])
        validity_masks.append(mask[:wh, :ww])
    return warped_imgs, validity_masks


def _compute_seam_masks(
    warped_imgs: list[np.ndarray],
    validity_masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    warped_sizes: list[tuple[int, int]],
    out_w: int,
    out_h: int,
) -> list[np.ndarray]:
    """
    Run graph-cut seam finding at low resolution, then resize results to full
    resolution and place on canvas-sized arrays.
    """
    seam_finder = SeamFinder("gc_color")

    # Compute scale to run seam finding at SEAM_MEGAPIX
    full_px = warped_imgs[0].shape[0] * warped_imgs[0].shape[1]
    seam_scale = min(1.0, math.sqrt(SEAM_MEGAPIX * 1e6 / full_px))

    warped_low = [cv.resize(w, None, fx=seam_scale, fy=seam_scale) for w in warped_imgs]
    masks_low = [cv.resize(m, None, fx=seam_scale, fy=seam_scale) for m in validity_masks]
    corners_low = [(int(cx * seam_scale), int(cy * seam_scale)) for (cx, cy) in corners]

    seam_masks_low = seam_finder.find(warped_low, corners_low, masks_low)
    # SeamFinder may return cv2.UMat — convert to numpy
    seam_masks_low = [
        cv.UMat.get(m) if isinstance(m, cv.UMat) else np.asarray(m)
        for m in seam_masks_low
    ]

    canvas_seam_masks = []
    for seam_mask_low, validity_mask, (cx, cy), (ww, wh) in zip(
        seam_masks_low, validity_masks, corners, warped_sizes
    ):
        seam_mask_full = SeamFinder.resize(seam_mask_low, validity_mask[:wh, :ww])

        canvas_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        canvas_mask[cy:cy + wh, cx:cx + ww] = seam_mask_full
        canvas_seam_masks.append(canvas_mask)

    return canvas_seam_masks


def _compute_exposure_gains(
    warped_imgs: list[np.ndarray],
    validity_masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    warped_sizes: list[tuple[int, int]],
) -> list[np.ndarray]:
    """
    Run gain exposure compensation at low resolution, apply it to calibration
    frames, and return per-channel gain factors (shape (3,)) per camera.
    """
    # Run compensator at low res for speed
    full_px = warped_imgs[0].shape[0] * warped_imgs[0].shape[1]
    comp_scale = min(1.0, math.sqrt(0.1e6 / full_px))

    warped_low = [cv.resize(w, None, fx=comp_scale, fy=comp_scale) for w in warped_imgs]
    masks_low = [cv.resize(m, None, fx=comp_scale, fy=comp_scale) for m in validity_masks]
    corners_low = [(int(cx * comp_scale), int(cy * comp_scale)) for (cx, cy) in corners]

    compensator = cv.detail.ExposureCompensator_createDefault(
        cv.detail.ExposureCompensator_GAIN
    )
    compensator.feed(corners_low, warped_low, masks_low)

    gains = []
    for i, (warped, mask) in enumerate(zip(warped_low, masks_low)):
        original = warped.copy()
        corrected = compensator.apply(i, corners_low[i], warped.copy(), mask)

        gain = np.ones(3, dtype=np.float32)
        valid = mask > 0
        if valid.any():
            for c in range(3):
                orig_mean = float(original[:, :, c][valid].mean())
                corr_mean = float(corrected[:, :, c][valid].mean())
                if orig_mean > 1.0:
                    gain[c] = corr_mean / orig_mean
        gains.append(gain)

    return gains


def calibrate(left: Path, right: Path) -> CalibrationResult:
    """
    Calibrate stitching for a left/right video pair.

    Samples the middle 30-second window first, then retries up to MAX_ATTEMPTS
    times with evenly-distributed windows. Raises RuntimeError if all attempts fail.
    """
    frame_w, frame_h, _, duration = probe_video(left)

    full_megapix = (frame_w * frame_h) / 1e6
    medium_scale = min(1.0, math.sqrt(MEDIUM_MEGAPIX / full_megapix))

    _DETECTOR_CONFIGS = [
        {"detector": "orb",   "nfeatures": 1000, "confidence_threshold": 0.1},
        {"detector": "orb",   "nfeatures": 3000, "confidence_threshold": 0.1},
        {"detector": "sift",  "nfeatures": 1000, "confidence_threshold": 0.1},
        {"detector": "akaze", "nfeatures": 1000, "confidence_threshold": 0.1},
        {"detector": "akaze", "nfeatures": 1000, "confidence_threshold": 0.01},
        {"detector": "sift",  "nfeatures": 3000, "confidence_threshold": 0.01},
    ]

    last_error: Exception | None = None

    for window_start in _window_starts(duration):
        sample_time = window_start + WINDOW_DURATION / 2
        frame_left = read_frame_at(left, sample_time, frame_w, frame_h)
        frame_right = read_frame_at(right, sample_time, frame_w, frame_h)
        if frame_left is None or frame_right is None:
            continue

        stitcher = None
        for cfg in _DETECTOR_CONFIGS:
            try:
                s = _CalibrationStitcher(**cfg)
                s.stitch([frame_left, frame_right])
                stitcher = s
                break
            except Exception as e:
                last_error = e

        if stitcher is None:
            continue

        cal = _build_remap_result(
            cameras=stitcher._cameras,
            scale=stitcher.warper.scale,
            warper_type=stitcher.warper.warper_type,
            frame_w=frame_w,
            frame_h=frame_h,
            confidence=stitcher._confidence,
            medium_scale=medium_scale,
        )

        warped_imgs, validity_masks = _warp_frames(
            [frame_left, frame_right], cal, frame_w, frame_h
        )

        try:
            cal.seam_masks = _compute_seam_masks(
                warped_imgs, validity_masks, cal.corners, cal.warped_sizes, cal.out_w, cal.out_h
            )
        except Exception as e:
            print(f"Warning: seam finding failed ({e}), falling back to distance-transform blend")
            cal.seam_masks = []

        try:
            cal.gains = _compute_exposure_gains(
                warped_imgs, validity_masks, cal.corners, cal.warped_sizes
            )
        except Exception as e:
            print(f"Warning: exposure compensation failed ({e}), skipping")
            cal.gains = []

        return cal

    raise RuntimeError(
        f"Calibration failed for {left.name} / {right.name}: {last_error}"
    )
