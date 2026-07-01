import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import correlate

SAMPLE_RATE = 8000
CONFIDENCE_THRESHOLD = 0.1


def _extract_audio(filepath: Path, max_seconds: float | None = None) -> np.ndarray | None:
    """Extract mono audio from a video file as a normalised float32 array at SAMPLE_RATE."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(filepath),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
    ]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "s16le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) == 0:
        return None

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak == 0:
        return None
    return audio / peak


def audio_offset(
    left: Path,
    right: Path,
    search_window_seconds: float | None = None,
) -> tuple[float, float] | None:
    """
    Compute the time offset between two videos using audio cross-correlation.

    Returns (offset_seconds, confidence) where:
    - offset_seconds > 0 means right audio starts that many seconds after left
    - offset_seconds < 0 means left audio starts that many seconds after right
    - confidence is the normalised correlation peak (0–1); below CONFIDENCE_THRESHOLD
      the result is unreliable

    Returns None if either file has no usable audio.

    If search_window_seconds is given, only searches within ±search_window_seconds
    of zero offset — use this to bound the search when you have a rough timestamp
    estimate to avoid false peaks on long recordings.
    """
    # Only extract as much audio as needed to cover the search window.
    # 3× the window gives enough context on either side; unconstrained extracts
    # the whole file which is very slow on long recordings.
    extract_seconds = search_window_seconds * 3 if search_window_seconds is not None else None
    left_audio = _extract_audio(left, extract_seconds)
    right_audio = _extract_audio(right, extract_seconds)

    if left_audio is None or right_audio is None:
        return None

    correlation = correlate(left_audio, right_audio, mode="full", method="fft")

    center = len(right_audio) - 1  # zero-lag index in full correlation output

    if search_window_seconds is not None:
        window = int(search_window_seconds * SAMPLE_RATE)
        lo = max(0, center - window)
        hi = min(len(correlation), center + window + 1)
        search = correlation[lo:hi]
        peak_idx = np.argmax(np.abs(search)) + lo
    else:
        peak_idx = np.argmax(np.abs(correlation))

    lag_samples = peak_idx - center
    offset_seconds = lag_samples / SAMPLE_RATE

    # Normalise confidence against theoretical maximum (dot product of both signals)
    max_possible = np.sqrt(np.sum(left_audio ** 2) * np.sum(right_audio ** 2))
    confidence = float(abs(correlation[peak_idx]) / max_possible) if max_possible > 0 else 0.0

    return offset_seconds, confidence
