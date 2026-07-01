"""
Export calibration JSON for a left/right video pair without stitching.

Usage:
    uv run python -m footage_studio.stitching.export_calibration LEFT RIGHT OUTPUT_DIR
"""

import json
import sys
from pathlib import Path

import static_ffmpeg

from footage_studio.stitching.calibration import calibrate


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: export_calibration LEFT RIGHT OUTPUT_DIR")
        sys.exit(1)

    left = Path(sys.argv[1])
    right = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    for p in (left, right):
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)

    static_ffmpeg.add_paths()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Calibrating {left.name} / {right.name} ...")
    cal = calibrate(left, right)
    print(f"Confidence: {cal.confidence:.3f}")

    calib_path = output_dir / "calibration.json"
    calib_path.write_text(json.dumps({
        "warper_scale": cal.warper_scale,
        "canvas_origin": list(cal.canvas_origin),
        "canvas_size": [cal.out_w, cal.out_h],
        "confidence": cal.confidence,
    }, indent=2))

    print(f"Written to {calib_path}")


if __name__ == "__main__":
    main()
