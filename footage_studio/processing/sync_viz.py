"""
Audio sync visualiser: plot both waveforms aligned at the computed offset,
plus the cross-correlation curve so you can judge peak quality visually.

Usage:
    uv run python -m footage_studio.processing.sync_viz LEFT RIGHT [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np

from footage_studio.processing.audio_sync import (
    CONFIDENCE_THRESHOLD,
    SAMPLE_RATE,
    _extract_audio,
    audio_offset,
)


def plot_sync(left: Path, right: Path, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise SystemExit("matplotlib is required: uv add matplotlib")

    print("Extracting audio...")
    left_audio = _extract_audio(left)
    right_audio = _extract_audio(right)
    if left_audio is None or right_audio is None:
        raise SystemExit("Could not extract audio from one or both files.")

    print("Computing cross-correlation...")
    result = audio_offset(left, right, search_window_seconds=30.0)
    if result is None:
        raise SystemExit("audio_offset returned None.")
    offset_s, confidence = result

    print(f"Offset: {offset_s:+.3f}s  Confidence: {confidence:.4f} (threshold: {CONFIDENCE_THRESHOLD})")
    accepted = confidence >= CONFIDENCE_THRESHOLD

    # --- downsample for plotting (keep every Nth sample so plots aren't huge) ---
    PLOT_RATE = 200  # samples/sec for waveform display
    step = max(1, SAMPLE_RATE // PLOT_RATE)

    # Build aligned time axes
    left_t = np.arange(len(left_audio)) / SAMPLE_RATE
    right_t = np.arange(len(right_audio)) / SAMPLE_RATE + offset_s  # shifted to align

    # Cross-correlation curve (full, search window ±30s)
    from scipy.signal import correlate
    correlation = correlate(left_audio, right_audio, mode="full", method="fft")
    center = len(right_audio) - 1
    window = int(30.0 * SAMPLE_RATE)
    lo = max(0, center - window)
    hi = min(len(correlation), center + window + 1)
    corr_slice = correlation[lo:hi]
    corr_lags = (np.arange(lo, hi) - center) / SAMPLE_RATE  # seconds

    # Normalise correlation for display
    max_possible = np.sqrt(np.sum(left_audio ** 2) * np.sum(right_audio ** 2))
    corr_norm = corr_slice / max_possible if max_possible > 0 else corr_slice

    # --- figure layout ---
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Audio Sync: {left.name} vs {right.name}\n"
        f"Offset: {offset_s:+.3f}s  |  Confidence: {confidence:.4f}  |  "
        f"{'ACCEPTED' if accepted else 'REJECTED (below threshold)'}",
        fontsize=11,
    )
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 2], hspace=0.45)

    # Top: left waveform
    ax_left = fig.add_subplot(gs[0])
    ax_left.plot(left_t[::step], left_audio[::step], lw=0.4, color="steelblue", label="Left")
    ax_left.set_ylabel("Amplitude")
    ax_left.set_xlabel("Time (s)")
    ax_left.legend(loc="upper right", fontsize=9)
    ax_left.set_xlim(left_t[0], left_t[-1])

    # Middle: both waveforms overlaid, right shifted to align
    ax_both = fig.add_subplot(gs[1])
    # Clip to shared window for clarity
    shared_start = max(left_t[0], right_t[0])
    shared_end = min(left_t[-1], right_t[-1])
    ax_both.plot(left_t[::step], left_audio[::step], lw=0.4, color="steelblue", alpha=0.7, label="Left")
    ax_both.plot(right_t[::step], right_audio[::step], lw=0.4, color="darkorange", alpha=0.7, label=f"Right (shifted {offset_s:+.3f}s)")
    ax_both.set_ylabel("Amplitude")
    ax_both.set_xlabel("Time (s, aligned)")
    ax_both.set_xlim(shared_start, shared_end)
    ax_both.legend(loc="upper right", fontsize=9)

    # Bottom: cross-correlation with peak marked
    ax_corr = fig.add_subplot(gs[2])
    ax_corr.plot(corr_lags, corr_norm, lw=0.6, color="seagreen")
    ax_corr.axvline(offset_s, color="red", lw=1.2, linestyle="--", label=f"Peak at {offset_s:+.3f}s")
    ax_corr.axhline(0, color="black", lw=0.4)
    ax_corr.set_ylabel("Normalised correlation")
    ax_corr.set_xlabel("Lag (s)  [positive = right starts after left]")
    ax_corr.legend(loc="upper right", fontsize=9)
    ax_corr.set_xlim(corr_lags[0], corr_lags[-1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise audio sync between two videos.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: sync_viz.png next to LEFT file)",
    )
    args = parser.parse_args()

    out = args.out or args.left.parent / "sync_viz.png"
    plot_sync(args.left, args.right, out)


if __name__ == "__main__":
    main()
