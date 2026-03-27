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


def read_frame_at(path: Path, time_s: float, width: int, height: int) -> np.ndarray | None:
    """Decode a single frame at the given timestamp. Returns BGR uint8 array or None."""
    try:
        out, _ = (
            ffmpeg
            .input(str(path), ss=time_s)
            .output("pipe:", vframes=1, format="rawvideo", pix_fmt="bgr24")
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error:
        return None
    expected = width * height * 3
    if len(out) < expected:
        return None
    return np.frombuffer(out, np.uint8)[:expected].reshape(height, width, 3).copy()


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
