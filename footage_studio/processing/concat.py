import subprocess
import tempfile
from pathlib import Path


def concatenate(filepaths: list[Path], output_path: Path, metadata: dict | None = None) -> None:
    """Concatenate video files using ffmpeg concat demuxer, optionally embedding metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for fp in filepaths:
            f.write(f"file '{fp}'\n")
        list_path = Path(f.name)

    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy"]
        if metadata:
            for key, value in metadata.items():
                cmd.extend(["-metadata", f"{key}={value}"])
        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        list_path.unlink()
