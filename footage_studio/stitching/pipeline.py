import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import Callable

from footage_studio.core import set_metadata
from footage_studio.processing import sync_trim
from footage_studio.stitching.calibration import CalibrationResult

# Set to True during development to keep source files after stitching
KEEP_SOURCES = True


def _blend_weights(
    H_warp: np.ndarray,
    offset: tuple[int, int],
    out_w: int,
    out_h: int,
    frame_h: int,
    frame_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute per-pixel blend weights for left and right cameras on the output canvas.
    Uses distance transform for smooth feathering in the overlap region.
    Returns (weight_left, weight_right) as float32 arrays of shape (out_h, out_w, 3).
    """
    mask_frame = np.ones((frame_h, frame_w), dtype=np.uint8) * 255

    # Coverage of left camera (warped onto canvas)
    warped_mask_left = cv2.warpPerspective(mask_frame, H_warp, (out_w, out_h))

    # Coverage of right camera (placed at offset)
    mask_right_canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    x_off, y_off = offset
    mask_right_canvas[y_off:y_off + frame_h, x_off:x_off + frame_w] = 255

    # Distance transform gives distance-to-edge within coverage → used as blend weight
    dist_left = cv2.distanceTransform((warped_mask_left > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_right = cv2.distanceTransform((mask_right_canvas > 0).astype(np.uint8), cv2.DIST_L2, 5)

    total = dist_left + dist_right
    total = np.where(total == 0, 1.0, total)  # avoid division by zero

    w_left = (dist_left / total).astype(np.float32)
    w_right = (dist_right / total).astype(np.float32)

    # Expand to 3 channels
    return (
        np.stack([w_left, w_left, w_left], axis=-1),
        np.stack([w_right, w_right, w_right], axis=-1),
    )


def stitch_session(
    left: Path,
    right: Path,
    output: Path,
    cal: CalibrationResult,
    progress_callback: Callable[[int, int], None] | None = None,
    delete_sources: bool = not KEEP_SOURCES,
) -> None:
    """
    Stitch a left/right video pair into a single panoramic video.

    Steps:
    1. Sync-trim both videos to their shared time window
    2. Precompute blend weight maps
    3. For each frame pair: warp left with H_warp, blend with right, pipe to ffmpeg
    4. Write PANORAMIC metadata to output
    5. Delete source files if delete_sources=True
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp_left = output.parent / "_tmp_left.mp4"
    tmp_right = output.parent / "_tmp_right.mp4"
    try:
        sync_trim(left, right, tmp_left, tmp_right)
        _stitch_trimmed(tmp_left, tmp_right, output, cal, progress_callback)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        tmp_left.unlink(missing_ok=True)
        tmp_right.unlink(missing_ok=True)

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
) -> None:
    cap_left = cv2.VideoCapture(str(left))
    cap_right = cv2.VideoCapture(str(right))
    try:
        fps = cap_left.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first = cap_left.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from left video")
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_h, frame_w = first.shape[:2]
        out_w, out_h = cal.out_w, cal.out_h
        x_off, y_off = cal.offset

        weight_left, weight_right = _blend_weights(
            cal.H_warp, cal.offset, out_w, out_h, frame_h, frame_w
        )

        # Pre-allocate canvas buffers to avoid per-frame allocation
        canvas_left = np.zeros((out_h, out_w, 3), dtype=np.float32)
        canvas_right = np.zeros((out_h, out_w, 3), dtype=np.float32)

        process = (
            ffmpeg
            .input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{out_w}x{out_h}", r=fps)
            .output(str(output), pix_fmt="yuv420p", vcodec="libx264", crf=18)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )

        try:
            frame_num = 0
            while True:
                ret_l, frame_left = cap_left.read()
                ret_r, frame_right = cap_right.read()
                if not ret_l or not ret_r:
                    break

                # Warp left frame onto canvas
                warped_left = cv2.warpPerspective(frame_left, cal.H_warp, (out_w, out_h))

                # Place both frames into float canvas buffers
                np.copyto(canvas_left, warped_left.astype(np.float32))
                canvas_right[:] = 0
                canvas_right[y_off:y_off + frame_h, x_off:x_off + frame_w] = \
                    frame_right.astype(np.float32)

                # Blend using precomputed weight maps
                result = (canvas_left * weight_left + canvas_right * weight_right).astype(np.uint8)

                process.stdin.write(result.tobytes())
                frame_num += 1
                if progress_callback:
                    progress_callback(frame_num, total_frames)
        finally:
            process.stdin.close()
            stderr_output = process.stderr.read().decode(errors="replace") if process.stderr else ""
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"ffmpeg exited with code {returncode}:\n{stderr_output[-2000:]}")
    finally:
        cap_left.release()
        cap_right.release()
