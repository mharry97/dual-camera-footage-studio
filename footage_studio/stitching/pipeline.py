import cv2
import ffmpeg
import numpy as np
import threading
from pathlib import Path
from typing import Callable

from footage_studio.core import set_metadata
from footage_studio.processing.trim import compute_sync_offsets
from footage_studio.stitching.calibration import CalibrationResult
from footage_studio.stitching.video_io import open_frame_reader, probe_video

# Set to True during development to keep source files after stitching
KEEP_SOURCES = True


def _precompute_blend_weights(
    cal: CalibrationResult,
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return float32 blend weight arrays (out_h, out_w) for cameras 0 and 1.

    If seam masks are available, uses hard-cut seam blending with a thin
    feather (~30px) near the seam boundary. Falls back to distance-transform
    feathering if seam masks are absent.
    """
    out_w, out_h = cal.out_w, cal.out_h

    if cal.seam_masks:
        # Feather-blend within a narrow band around the seam
        FEATHER_RADIUS = 30
        kernel = np.ones((FEATHER_RADIUS * 2 + 1, FEATHER_RADIUS * 2 + 1), np.float32)
        kernel /= kernel.sum()
        w0 = cv2.filter2D(
            (cal.seam_masks[0] > 0).astype(np.float32), -1, kernel
        ).clip(0, 1)
        w1 = 1.0 - w0
        return w0.astype(np.float32), w1.astype(np.float32)

    # Fallback: distance-transform feathering
    solid_mask = np.ones((frame_h, frame_w), dtype=np.uint8) * 255
    canvas_masks = []
    for (map_x, map_y), (cx, cy), (ww, wh) in zip(cal.maps, cal.corners, cal.warped_sizes):
        warped_mask = cv2.remap(solid_mask, map_x, map_y, cv2.INTER_NEAREST)
        canvas_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        canvas_mask[cy:cy + wh, cx:cx + ww] = warped_mask[:wh, :ww]
        canvas_masks.append(canvas_mask)

    dist_0 = cv2.distanceTransform((canvas_masks[0] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_1 = cv2.distanceTransform((canvas_masks[1] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    total = dist_0 + dist_1
    total = np.where(total == 0, 1.0, total)
    return (dist_0 / total).astype(np.float32), (dist_1 / total).astype(np.float32)


def _apply_gain(frame: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply per-channel exposure gain to a frame."""
    if np.allclose(gain, 1.0):
        return frame
    return np.clip(frame.astype(np.float32) * gain[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)


def stitch_session(
    left: Path,
    right: Path,
    output: Path,
    cal: CalibrationResult,
    progress_callback: Callable[[int, int], None] | None = None,
    delete_sources: bool = not KEEP_SOURCES,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        left_offset, right_offset, duration = compute_sync_offsets(
            left, right, log_path=output.parent / "sync_debug.log"
        )
        _stitch_trimmed(left, right, output, cal, progress_callback, left_offset, right_offset, duration)
    except Exception:
        output.unlink(missing_ok=True)
        raise

    set_metadata(output, "status", "PANORAMIC")

    if delete_sources:
        left.unlink(missing_ok=True)
        right.unlink(missing_ok=True)


def _stitch_trimmed(
    left: Path,
    right: Path,
    output: Path,
    cal: CalibrationResult,
    progress_callback: Callable[[int, int], None] | None,
    left_offset_s: float = 0.0,
    right_offset_s: float = 0.0,
    duration_s: float | None = None,
) -> None:
    frame_w, frame_h, fps, _ = probe_video(left)
    _, _, fps_right, _ = probe_video(right)
    if abs(fps - fps_right) > 0.1:
        import warnings
        warnings.warn(
            f"Frame rate mismatch: left={fps:.3f}fps, right={fps_right:.3f}fps"
            " — output may drift over time"
        )

    total_frames = int(duration_s * fps) if duration_s is not None else None
    out_w, out_h = cal.out_w, cal.out_h

    (map_x_0, map_y_0), (cx0, cy0), (ww0, wh0) = cal.maps[0], cal.corners[0], cal.warped_sizes[0]
    (map_x_1, map_y_1), (cx1, cy1), (ww1, wh1) = cal.maps[1], cal.corners[1], cal.warped_sizes[1]

    gain_0 = cal.gains[0] if cal.gains else np.ones(3, dtype=np.float32)
    gain_1 = cal.gains[1] if cal.gains else np.ones(3, dtype=np.float32)

    w0, w1 = _precompute_blend_weights(cal, frame_w, frame_h)
    w0_3ch = np.stack([w0, w0, w0], axis=-1)
    w1_3ch = np.stack([w1, w1, w1], axis=-1)

    canvas_0 = np.zeros((out_h, out_w, 3), dtype=np.float32)
    canvas_1 = np.zeros((out_h, out_w, 3), dtype=np.float32)

    encode_process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{out_w}x{out_h}", r=fps)
        .output(str(output), pix_fmt="yuv420p", vcodec="libx264", crf=23, movflags="+faststart")
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    # Drain stderr in a background thread to prevent pipe buffer deadlock
    # (ffmpeg writes progress every ~0.5s; without draining it fills the 64KB OS buffer)
    stderr_chunks: list[bytes] = []
    def _drain_stderr():
        for chunk in iter(lambda: encode_process.stderr.read(4096), b""):
            stderr_chunks.append(chunk)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        with open_frame_reader(left, left_offset_s, frame_w, frame_h) as read_left, \
             open_frame_reader(right, right_offset_s, frame_w, frame_h) as read_right:
            frame_num = 0
            while total_frames is None or frame_num < total_frames:
                frame_left = read_left()
                frame_right = read_right()
                if frame_left is None or frame_right is None:
                    break

                frame_left = _apply_gain(frame_left, gain_0)
                frame_right = _apply_gain(frame_right, gain_1)

                warped_0 = cv2.remap(frame_left, map_x_0, map_y_0, cv2.INTER_LINEAR)
                warped_1 = cv2.remap(frame_right, map_x_1, map_y_1, cv2.INTER_LINEAR)

                canvas_0[:] = 0
                canvas_1[:] = 0
                canvas_0[cy0:cy0 + wh0, cx0:cx0 + ww0] = warped_0[:wh0, :ww0].astype(np.float32)
                canvas_1[cy1:cy1 + wh1, cx1:cx1 + ww1] = warped_1[:wh1, :ww1].astype(np.float32)

                result = (canvas_0 * w0_3ch + canvas_1 * w1_3ch).astype(np.uint8)

                encode_process.stdin.write(result.tobytes())
                frame_num += 1
                if progress_callback:
                    progress_callback(frame_num, total_frames or frame_num)
    finally:
        encode_process.stdin.close()
        stderr_thread.join()
        stderr_output = b"".join(stderr_chunks).decode(errors="replace")
        returncode = encode_process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode}:\n{stderr_output[-2000:]}")
