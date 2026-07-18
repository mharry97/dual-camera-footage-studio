"""
One-off video stabilisation via ffmpeg's two-pass vidstab filters.

Pass 1 (vidstabdetect) analyses camera motion and writes a transforms file;
pass 2 (vidstabtransform) applies smoothed compensation. Audio is stream-copied.

Usage:
    uv run python -m footage_studio.processing.stabilise INPUT [-o OUTPUT]
        [--smoothing FRAMES] [--shakiness 1-10] [--unsharp]

Output defaults to {input stem}_stabilised.mp4 next to the input.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import static_ffmpeg


def stabilise(
    input_path: Path,
    output_path: Path,
    smoothing: int = 30,
    shakiness: int = 5,
    unsharp: bool = False,
) -> None:
    static_ffmpeg.add_paths()
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin, "ffmpeg binary not found after static_ffmpeg.add_paths()"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        trf_path = Path(tmp_str) / "transforms.trf"

        print(f"Pass 1/2: analysing motion in {input_path.name}...")
        result = subprocess.run(
            [
                ffmpeg_bin, "-y",
                "-i", str(input_path),
                "-vf", f"vidstabdetect=shakiness={shakiness}:result={trf_path}",
                "-f", "null", "-",
            ],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"vidstabdetect pass failed (exit code {result.returncode})"
            )

        transform_filter = (
            f"vidstabtransform=input={trf_path}:smoothing={smoothing}:crop=black"
        )
        if unsharp:
            transform_filter += ",unsharp=5:5:0.8:3:3:0.4"

        print(f"Pass 2/2: rendering stabilised video to {output_path.name}...")
        result = subprocess.run(
            [
                ffmpeg_bin, "-y",
                "-i", str(input_path),
                "-vf", transform_filter,
                "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"vidstabtransform pass failed (exit code {result.returncode})"
            )

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Two-pass vidstab stabilisation")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: {input stem}_stabilised.mp4 next to input)",
    )
    parser.add_argument(
        "--smoothing", type=int, default=30,
        help="Frames of motion smoothing; higher = steadier (default: 30)",
    )
    parser.add_argument("--shakiness", type=int, default=5, choices=range(1, 11),
                        help="How shaky the footage is, 1-10 (default: 5)")
    parser.add_argument("--unsharp", action="store_true",
                        help="Apply a light unsharp filter after stabilisation")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_name(f"{args.input.stem}_stabilised.mp4")
    stabilise(args.input, output, args.smoothing, args.shakiness, args.unsharp)


if __name__ == "__main__":
    main()
