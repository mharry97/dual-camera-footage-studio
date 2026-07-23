import json
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
from footage_studio.stitching.video_io import has_audio_stream, probe_video

TAIL_THRESHOLD_S = 1.0

# Stitched output is encoded directly at mobile-compatible resolution — the
# full-res canvas is never written to disk. The original raw clips already
# archived under _processed/ are the archive of record, so synced sources are
# deleted after a successful stitch.
MOBILE_MAX_W = 3840


def _output_dims(cal: "CalibrationResult") -> tuple[int, int]:
    """Final output dimensions: canvas downscaled to MOBILE_MAX_W, height even."""
    if cal.out_w <= MOBILE_MAX_W:
        return cal.out_w, cal.out_h
    out_h = round(cal.out_h * MOBILE_MAX_W / cal.out_w / 2) * 2
    return MOBILE_MAX_W, out_h


def _precompute_blend_weights(
    cal: CalibrationResult,
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

    # Fallback: distance-transform feathering using per-camera frame sizes
    canvas_masks = []
    for i, ((map_x, map_y), (cx, cy), (ww, wh)) in enumerate(
        zip(cal.maps, cal.corners, cal.warped_sizes)
    ):
        fw, fh = cal.frame_sizes[i] if i < len(cal.frame_sizes) else (map_x.shape[1], map_x.shape[0])
        solid_mask = np.ones((fh, fw), dtype=np.uint8) * 255
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


def make_mobile_copy(
    source: Path,
    output: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Transcode a full-res stitched file to a mobile-compatible H.264 copy."""
    static_ffmpeg.add_paths()
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin, "ffmpeg binary not found after static_ffmpeg.add_paths()"

    _, _, fps, duration = probe_video(source)
    total_frames = int(duration * fps) if duration else None

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(source),
        "-vf", "scale='min(3840,iw)':-2",
        "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-level:v", "5.2", "-profile:v", "High",
        "-acodec", "copy",
        "-movflags", "+faststart",
        str(output),
    ]

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    stderr_chunks: list[bytes] = []

    def _drain() -> None:
        buf = b""
        assert process.stderr is not None
        for chunk in iter(lambda: process.stderr.read(256), b""):
            stderr_chunks.append(chunk)
            if progress_callback:
                buf += chunk
                for m in re.finditer(rb"frame=\s*(\d+)", buf):
                    try:
                        progress_callback(int(m.group(1)), total_frames or 0)
                    except ValueError:
                        pass
                last = buf.rfind(b"frame=")
                buf = buf[last:] if last >= 0 else buf[-64:]

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    t.join()
    returncode = process.wait()
    if returncode != 0:
        stderr_output = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with code {returncode}:\n{stderr_output[-2000:]}")


def _append_tail(
    stitched: Path,
    tail_source: Path,
    tail_start_s: float,
    out_w: int,
    out_h: int,
    on_left: bool,
    src_w: int,
    src_h: int,
    include_audio: bool = False,
) -> None:
    """Encode the single-camera tail and concat it onto the end of the stitched file.

    tail_source is always the camera whose audio was used for the main segment
    (it's the longer of the two by construction), so continuing to pull audio
    from it here just extends that same continuous track — no splice needed.
    """
    static_ffmpeg.add_paths()
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin

    scaled_w = int(round(src_w * out_h / src_h))
    scaled_w += scaled_w % 2
    x_offset = 0 if on_left else out_w - scaled_w

    merging = stitched.with_name(stitched.stem + ".merging.mp4")
    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            tail_path = tmp_dir / "tail.mp4"

            cmd = [
                ffmpeg_bin, "-y",
                "-ss", str(tail_start_s), "-i", str(tail_source),
                "-vf", f"scale={scaled_w}:{out_h},pad={out_w}:{out_h}:{x_offset}:0:black",
                "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
                "-level:v", "5.2", "-profile:v", "High", "-pix_fmt", "yuv420p",
            ]
            cmd += ["-acodec", "aac", "-b:a", "192k"] if include_audio else ["-an"]
            cmd += [str(tail_path)]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Tail encode failed: {result.stderr.decode(errors='replace')[-2000:]}"
                )

            concat_list = tmp_dir / "concat.txt"
            concat_list.write_text(
                f"file '{stitched.resolve()}'\nfile '{tail_path.resolve()}'\n"
            )
            result = subprocess.run(
                [
                    ffmpeg_bin, "-y",
                    "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-c", "copy", "-movflags", "+faststart",
                    str(merging),
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Tail concat failed: {result.stderr.decode(errors='replace')[-2000:]}"
                )

        merging.replace(stitched)
    except Exception:
        merging.unlink(missing_ok=True)
        raise


def stitch_session(
    left: Path,
    right: Path,
    output: Path,
    cal: CalibrationResult,
    progress_callback: Callable[[int, int], None] | None = None,
    delete_sources: bool = True,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    _, _, _, left_duration = probe_video(left)
    _, _, _, right_duration = probe_video(right)
    shared_duration = min(left_duration, right_duration)

    # The longer camera's audio spans the whole output timeline (main segment
    # plus any tail), so it's used throughout rather than switching tracks —
    # one continuous recording, no splice at the seam.
    audio_is_left = left_duration >= right_duration
    audio_source = left if audio_is_left else right
    audio_input_idx = 0 if audio_is_left else 1
    has_audio = has_audio_stream(audio_source)

    try:
        _stitch_ffmpeg(
            left, right, output, cal, progress_callback, 0.0, 0.0, shared_duration,
            audio_input_idx=audio_input_idx if has_audio else None,
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise

    out_w, out_h = _output_dims(cal)

    tail_duration = abs(left_duration - right_duration)
    if tail_duration > TAIL_THRESHOLD_S:
        tail_is_left = left_duration > right_duration
        tail_source = left if tail_is_left else right
        input_idx = 0 if tail_is_left else 1
        other_idx = 1 - input_idx
        src_w, src_h = cal.frame_sizes[input_idx]
        on_left = cal.corners[input_idx][0] <= cal.corners[other_idx][0]
        _append_tail(
            output, tail_source, shared_duration, out_w, out_h, on_left, src_w, src_h,
            include_audio=has_audio,
        )

    # Calibration is expressed in canvas pixels; the maths is linear, so scaling
    # every field by the output/canvas ratio keeps downstream spherical
    # projection consistent with the downscaled video.
    sx = out_w / cal.out_w
    sy = out_h / cal.out_h
    calib_path = output.parent / "calibration.json"
    calib_path.write_text(json.dumps({
        "warper_scale": cal.warper_scale * sx,
        "canvas_origin": [round(cal.canvas_origin[0] * sx), round(cal.canvas_origin[1] * sy)],
        "canvas_size": [out_w, out_h],
    }, indent=2))

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
    audio_input_idx: int | None = None,
) -> None:
    """
    Stitch using an ffmpeg filter graph — no Python frame loop.

    Pipeline:
      left.mp4 ──[colorchannelmixer]──[remap(xmap0,ymap0)]──┐
                                                              ├──[alphamerge+overlay]──[scale ≤3840w]──► output.mp4
      right.mp4 ──[colorchannelmixer]──[remap(xmap1,ymap1)]──┘
                                        ↑
                              weight0.png (cam0 blend weight as alpha)

    The blend runs at full canvas resolution; only the encoded output is
    downscaled to mobile width (single-generation encode — no separate
    full-res master is written).

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

        w0, _ = _precompute_blend_weights(cal)
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
        target_w, target_h = _output_dims(cal)
        if (target_w, target_h) != (out_w, out_h):
            overlay_out = (f"overlay=format=yuv420:shortest=1[ov];"
                           f"[ov]scale={target_w}:{target_h}[out]")
        else:
            overlay_out = "overlay=format=yuv420:shortest=1[out]"

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
        if audio_input_idx is not None:
            cmd += ["-map", f"{audio_input_idx}:a:0"]
        cmd += ["-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
                "-level:v", "5.2", "-profile:v", "High", "-movflags", "+faststart"]
        if audio_input_idx is not None:
            cmd += ["-acodec", "aac", "-b:a", "192k", "-shortest"]
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
