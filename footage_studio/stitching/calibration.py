import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.3
MAX_ATTEMPTS = 5
WINDOW_DURATION = 30.0  # seconds


@dataclass
class CalibrationResult:
    """
    Stores the computed homography and canvas layout for a left/right video pair.
    H maps points in the left frame to points in the right frame's coordinate space.
    H_warp = T @ H is the adjusted homography applied to every left frame during stitching.
    The right frame is placed at `offset` (x, y) on the output canvas.
    """
    H_warp: np.ndarray       # 3x3: warp applied to left frames
    offset: tuple[int, int]  # (x, y): where right frame is placed on canvas
    out_w: int
    out_h: int
    confidence: float

    def save(self, path: Path) -> None:
        np.savez(
            path,
            H_warp=self.H_warp,
            offset=np.array(self.offset),
            out_size=np.array([self.out_w, self.out_h]),
            confidence=np.array(self.confidence),
        )

    @classmethod
    def load(cls, path: Path) -> "CalibrationResult":
        data = np.load(str(path))
        return cls(
            H_warp=data["H_warp"],
            offset=tuple(data["offset"].tolist()),
            out_w=int(data["out_size"][0]),
            out_h=int(data["out_size"][1]),
            confidence=float(data["confidence"]),
        )


def _window_starts(duration: float) -> list[float]:
    """Return MAX_ATTEMPTS evenly-distributed window start times, middle section first."""
    section = duration / MAX_ATTEMPTS
    centers = [section * (i + 0.5) for i in range(MAX_ATTEMPTS)]
    starts = [max(0.0, c - WINDOW_DURATION / 2) for c in centers]
    mid = MAX_ATTEMPTS // 2
    order = [mid] + [i for i in range(MAX_ATTEMPTS) if i != mid]
    return [starts[i] for i in order]


def _read_frame(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    return frame if ret else None


def _try_calibrate(left_frame: np.ndarray, right_frame: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Detect ORB features, match with BFMatcher + Lowe ratio test, compute homography
    with RANSAC. Returns (H, confidence) where confidence = inlier_ratio.
    H maps left-frame coords to right-frame coords.
    """
    detector = cv2.ORB.create(nfeatures=2000)
    kp0, desc0 = detector.detectAndCompute(left_frame, None)
    kp1, desc1 = detector.detectAndCompute(right_frame, None)

    if desc0 is None or desc1 is None or len(kp0) < 8 or len(kp1) < 8:
        return None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(desc0, desc1, k=2)

    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0.0

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, 0.0

    confidence = float(mask.sum()) / len(good)
    return H, confidence


def _canvas_layout(H: np.ndarray, h: int, w: int) -> tuple[np.ndarray, tuple[int, int], int, int]:
    """
    Given H (left→right) and frame size, compute:
    - H_warp: adjusted homography to apply to left frames on the output canvas
    - offset (x, y): where the right frame sits on the canvas
    - out_w, out_h: canvas dimensions
    """
    corners_left = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_left, H)

    corners_right = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    all_corners = np.vstack([warped_corners, corners_right])

    x_min = float(np.floor(all_corners[:, :, 0].min()))
    y_min = float(np.floor(all_corners[:, :, 1].min()))
    x_max = float(np.ceil(all_corners[:, :, 0].max()))
    y_max = float(np.ceil(all_corners[:, :, 1].max()))

    # libx264 requires dimensions divisible by 2
    out_w = int(x_max - x_min)
    out_h = int(y_max - y_min)
    out_w += out_w % 2
    out_h += out_h % 2

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    H_warp = T @ H
    offset = (int(-x_min), int(-y_min))

    return H_warp, offset, out_w, out_h


def calibrate(left: Path, right: Path) -> CalibrationResult:
    """
    Calibrate stitching for a left/right video pair. Samples the middle 30-second
    window first, then retries up to MAX_ATTEMPTS times with evenly-distributed
    windows. Returns the best result found (even if below confidence threshold).
    """
    cap_left = cv2.VideoCapture(str(left))
    cap_right = cv2.VideoCapture(str(right))
    try:
        fps = cap_left.get(cv2.CAP_PROP_FPS)
        total_frames = cap_left.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0.0
        if duration < WINDOW_DURATION:
            raise ValueError(f"Video too short for calibration ({duration:.1f}s): {left.name}")

        best: CalibrationResult | None = None

        for window_start in _window_starts(duration):
            sample_time = window_start + WINDOW_DURATION / 2
            left_frame = _read_frame(cap_left, sample_time)
            right_frame = _read_frame(cap_right, sample_time)
            if left_frame is None or right_frame is None:
                continue

            H, confidence = _try_calibrate(left_frame, right_frame)
            if H is None:
                continue

            h, w = left_frame.shape[:2]
            H_warp, offset, out_w, out_h = _canvas_layout(H, h, w)

            # Reject degenerate homographies that produce unreasonably large canvases
            if out_w > w * 4 or out_h > h * 4:
                continue

            result = CalibrationResult(
                H_warp=H_warp, offset=offset,
                out_w=out_w, out_h=out_h,
                confidence=confidence,
            )
            if best is None or confidence > best.confidence:
                best = result
            if confidence >= CONFIDENCE_THRESHOLD:
                break

        if best is None:
            raise RuntimeError(f"Feature matching failed for {left.name} / {right.name}")

        return best
    finally:
        cap_left.release()
        cap_right.release()
