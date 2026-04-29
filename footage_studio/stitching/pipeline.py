import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import static_ffmpeg

from footage_studio.stitching.calibration import CalibrationResult
from footage_studio.stitching.video_io import probe_video

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
    """Apply per-channel exposure gain to a frame (BGR order)."""
    if np.allclose(gain, 1.0):
        return frame
    return np.clip(frame.astype(np.float32) * gain[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)


def _save_canvas_maps(cal: CalibrationResult, tmp_dir: Path) -> list[tuple[Path, Path]]:
    """
    Write canvas-sized uint16 remap tables as raw gray16le binary files.

    Each map file is out_w × out_h uint16 values (row-major). Values are
    source pixel coordinates. Pixels outside the warped region are set to
    65535 (out-of-range → ffmpeg remap fills them with black).
    """
    out_w, out_h = cal.out_w, cal.out_h
    INVALID = np.uint16(65535)
    paths = []
    for i, ((map_x, map_y), (cx, cy), (ww, wh)) in enumerate(
        zip(cal.maps, cal.corners, cal.warped_sizes)
    ):
        canvas_x = np.full((out_h, out_w), INVALID, dtype=np.uint16)
        canvas_y = np.full((out_h, out_w), INVALID, dtype=np.uint16)
        h_clip = min(wh, map_x.shape[0])
        w_clip = min(ww, map_x.shape[1])
        mx = np.clip(np.round(map_x[:h_clip, :w_clip]), 0, 65534).astype(np.uint16)
        my = np.clip(np.round(map_y[:h_clip, :w_clip]), 0, 65534).astype(np.uint16)
        canvas_x[cy:cy + h_clip, cx:cx + w_clip] = mx
        canvas_y[cy:cy + h_clip, cx:cx + w_clip] = my
        xp = tmp_dir / f"xmap{i}.raw"
        yp = tmp_dir / f"ymap{i}.raw"
        canvas_x.tofile(str(xp))
        canvas_y.tofile(str(yp))
        paths.append((xp, yp))
    return paths


def stitch_session(
    left: Path,
    right: Path,
    output: Path,
    cal: CalibrationResult,
    progress_callback: Callable[[int, int], None] | None = None,
    delete_sources: bool = not KEEP_SOURCES,
    mobile_friendly: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    # Inputs are assumed to be already synced by the manual sync step.
    # Re-running audio cross-correlation here produces spurious offsets on
    # pre-synced files. Use the shorter of the two durations instead.
    _, _, _, left_duration = probe_video(left)
    _, _, _, right_duration = probe_video(right)
    duration = min(left_duration, right_duration)

    try:
        _stitch_ffmpeg(left, right, output, cal, progress_callback, 0.0, 0.0, duration, mobile_friendly)
    except Exception:
        output.unlink(missing_ok=True)
        raise

    if delete_sources:
        left.unlink(missing_ok=True)
        right.unlink(missing_ok=True)


def _stitch_ffmpeg(
    left: Path,
    right: Path,
    output: Path,
    cal: CalibrationResult,
    progress_callback: Callable[[int, int], None] | None,
    left_offset_s: float = 0.0,
    right_offset_s: float = 0.0,
    duration_s: float | None = None,
    mobile_friendly: bool = False,
) -> None:
    """
    Stitch using an ffmpeg filter graph — no Python frame loop.

    Pipeline:
      left.mp4 ──[colorchannelmixer]──[remap(xmap0,ymap0)]──┐
                                                              ├──[alphamerge+overlay]──► output.mp4
      right.mp4 ──[colorchannelmixer]──[remap(xmap1,ymap1)]──┘
                                        ↑
                              weight0.png (cam0 blend weight as alpha)

    Map files are canvas-sized gray16le uint16 raw binary — ffmpeg remap
    filter expects integer pixel coordinates in this format.
    """
    static_ffmpeg.add_paths()
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin, "ffmpeg binary not found after static_ffmpeg.add_paths()"

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

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        map_paths = _save_canvas_maps(cal, tmp_dir)

        w0, _ = _precompute_blend_weights(cal, frame_w, frame_h)
        weight_path = tmp_dir / "weight0.png"
        cv2.imwrite(str(weight_path), (w0 * 255).astype(np.uint8))

        gain_0 = cal.gains[0] if cal.gains else np.ones(3, dtype=np.float32)
        gain_1 = cal.gains[1] if cal.gains else np.ones(3, dtype=np.float32)

        def _gain_filter(gains: np.ndarray) -> str:
            # gains are BGR (OpenCV); colorchannelmixer works in RGB after format=rgb24
            # rgb24 channel order: R=index0, G=index1, B=index2
            # gains BGR: gains[2]=R, gains[1]=G, gains[0]=B
            if np.allclose(gains, 1.0):
                return "format=rgb24"
            r, g, b = float(gains[2]), float(gains[1]), float(gains[0])
            return f"format=rgb24,colorchannelmixer=rr={r:.6f}:gg={g:.6f}:bb={b:.6f}"

        map_size = f"{out_w}x{out_h}"
        fps_str = f"{fps:.6f}"

        # Input indices:
        # 0 = left video, 1 = right video
        # 2 = xmap0, 3 = ymap0, 4 = xmap1, 5 = ymap1
        # 6 = weight0.png
        # Mobile-friendly: scale to max 3840px wide so libx264 stays at Level 5.2
        # rather than Level 6.0, which most mobile hardware decoders can't handle.
        overlay_out = "overlay=format=yuv420:shortest=1,scale='min(3840,iw)':-2[out]" \
            if mobile_friendly else "overlay=format=yuv420:shortest=1[out]"

        filter_complex = ";".join([
            f"[0]{_gain_filter(gain_0)}[lg]",
            f"[1]{_gain_filter(gain_1)}[rg]",
            "[lg][2][3]remap[w0]",
            "[rg][4][5]remap[w1]",
            "[w0]format=rgba[w0a]",
            "[w0a][6]alphamerge[w0_alpha]",
            f"[w1][w0_alpha]{overlay_out}",
        ])

        map_loop_args = [
            "-stream_loop", "-1",
            "-f", "rawvideo",
            "-pix_fmt", "gray16le",
            "-s", map_size,
            "-r", fps_str,
        ]

        cmd = [ffmpeg_bin, "-y"]
        # Left video
        cmd += ["-hwaccel", "auto"]
        cmd += ["-ss", str(left_offset_s)]
        if duration_s is not None:
            cmd += ["-t", str(duration_s)]
        cmd += ["-i", str(left)]
        # Right video
        cmd += ["-hwaccel", "auto"]
        cmd += ["-ss", str(right_offset_s)]
        if duration_s is not None:
            cmd += ["-t", str(duration_s)]
        cmd += ["-i", str(right)]
        # Map files (looped single-frame rawvideo)
        (xmap0, ymap0), (xmap1, ymap1) = map_paths[0], map_paths[1]
        cmd += map_loop_args + ["-i", str(xmap0)]
        cmd += map_loop_args + ["-i", str(ymap0)]
        cmd += map_loop_args + ["-i", str(xmap1)]
        cmd += map_loop_args + ["-i", str(ymap1)]
        # Weight image (looped)
        cmd += ["-loop", "1", "-i", str(weight_path)]

        cmd += ["-filter_complex", filter_complex, "-map", "[out]"]
        cmd += ["-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", "-movflags", "+faststart"]
        cmd += [str(output)]

        process = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        # Drain stderr in background, parse frame counts for progress callback
        stderr_chunks: list[bytes] = []

        def _drain_stderr() -> None:
            buf = b""
            assert process.stderr is not None
            for chunk in iter(lambda: process.stderr.read(256), b""):  # type: ignore[union-attr]
                stderr_chunks.append(chunk)
                if progress_callback:
                    buf += chunk
                    for m in re.finditer(rb"frame=\s*(\d+)", buf):
                        try:
                            frame_num = int(m.group(1))
                            progress_callback(frame_num, total_frames or frame_num)
                        except ValueError:
                            pass
                    # Keep only unparsed tail to avoid quadratic growth
                    last = buf.rfind(b"frame=")
                    buf = buf[last:] if last >= 0 else buf[-64:]

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()
        stderr_thread.join()

        returncode = process.wait()
        if returncode != 0:
            stderr_output = b"".join(stderr_chunks).decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg exited with code {returncode}:\n{stderr_output[-2000:]}"
            )
