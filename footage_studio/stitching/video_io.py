"""
ffmpeg-based video reading utilities.
cv2.VideoCapture cannot decode HEVC (H.265), so we use ffmpeg pipes for all decoding.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Generator

import ffmpeg
import numpy as np


def probe_video(path: Path) -> tuple[int, int, float, float]:
    """Return (width, height, fps, duration) for a video file."""
    probe = ffmpeg.probe(str(path))
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(vs["width"])
    height = int(vs["height"])
    num, den = vs["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(probe["format"]["duration"])
    return width, height, fps, duration


def has_audio_stream(path: Path) -> bool:
    """Return True if the file has at least one audio stream."""
    probe = ffmpeg.probe(str(path))
    return any(s["codec_type"] == "audio" for s in probe["streams"])


def read_frame_at(path: Path, time_s: float, width: int, height: int) -> np.ndarray | None:
    """Decode a single frame at the given timestamp. Returns BGR uint8 array or None."""
    try:
        out, _ = (
            ffmpeg
            .input(str(path), ss=time_s)
            .output("pipe:", vframes=1, format="rawvideo", pix_fmt="bgr24")
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"[read_frame_at] ffmpeg error at t={time_s:.1f}s for {path.name}:\n"
              f"{e.stderr.decode(errors='replace')[-500:]}")
        return None
    expected = width * height * 3
    if len(out) < expected:
        print(f"[read_frame_at] short output at t={time_s:.1f}s for {path.name}: "
              f"got {len(out)} bytes, expected {expected}")
        return None
    return np.frombuffer(out, np.uint8)[:expected].reshape(height, width, 3).copy()


def actual_frame_size(path: Path, probed_width: int, probed_height: int) -> tuple[int, int]:
    """
    Decode one frame to get true decoded dimensions.

    Some DJI H.265 files report 4:3 display dimensions in container metadata
    while the actual H.265 bitstream encodes 16:9 frames (conformance crop).
    ffprobe returns the container dimensions; this returns what ffmpeg decodes.
    """
    try:
        out, _ = (
            ffmpeg
            .input(str(path), ss=1.0)
            .output("pipe:", vframes=1, format="rawvideo", pix_fmt="bgr24")
            .run(capture_stdout=True, capture_stderr=True)
        )
        n = len(out)
        if n > 0 and probed_width > 0 and n % (probed_width * 3) == 0:
            return probed_width, n // (probed_width * 3)
    except ffmpeg.Error:
        pass
    return probed_width, probed_height


@contextlib.contextmanager
def open_frame_reader(
    path: Path, start_s: float, width: int, height: int
) -> Generator[object, None, None]:
    """
    Context manager that yields a callable `read() -> np.ndarray | None`
    for sequential frame reading starting at start_s.
    """
    process = (
        ffmpeg
        .input(str(path), ss=start_s)
        .output("pipe:", format="rawvideo", pix_fmt="bgr24")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3

    def read() -> np.ndarray | None:
        raw = process.stdout.read(frame_size)
        if len(raw) < frame_size:
            return None
        return np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()

    try:
        yield read
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()
