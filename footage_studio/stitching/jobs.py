import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from footage_studio.stitching.calibration import CalibrationResult, CONFIDENCE_THRESHOLD


@dataclass
class JobProgress:
    stage: str           # "calibrating" | "stitching" | "transcoding" | "done" | "failed"
    session_index: int   # 0-based index of current session
    total_sessions: int
    current_frame: int = 0
    total_frames: int = 0
    eta_seconds: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class StitchSession:
    name: str
    left_path: str
    right_path: str


@dataclass
class Job:
    id: str
    progress: JobProgress
    sessions: list[StitchSession]


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def start_stitch_job(sessions: list[StitchSession], output_dir: Path) -> str:
    job_id = str(uuid.uuid4())[:8]
    job = Job(
        id=job_id,
        progress=JobProgress(stage="calibrating", session_index=0, total_sessions=len(sessions)),
        sessions=sessions,
    )
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job, output_dir), daemon=True)
    thread.start()
    return job_id


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def _run_job(job: Job, output_dir: Path) -> None:
    from wakepy import keep

    from footage_studio.stitching.calibration import calibrate
    from footage_studio.stitching.pipeline import make_mobile_copy, stitch_session

    n = len(job.sessions)
    warnings: list[str] = []

    with keep.running(on_fail="pass"):
        # Phase 1: calibrate all sessions up front
        calibrations: list[tuple[CalibrationResult, StitchSession]] = []
        for i, session in enumerate(job.sessions):
            job.progress = JobProgress(stage="calibrating", session_index=i, total_sessions=n)
            try:
                cal = calibrate(Path(session.left_path), Path(session.right_path))
            except Exception as e:
                job.progress = JobProgress(
                    stage="failed", session_index=i, total_sessions=n, error=str(e)
                )
                return

            if cal.confidence < CONFIDENCE_THRESHOLD:
                warnings.append(
                    f"Low calibration confidence ({cal.confidence:.2f}) for {session.name} "
                    f"— stitching may have visible seams."
                )
            calibrations.append((cal, session))

        # Phase 2: stitch each session
        for i, (cal, session) in enumerate(calibrations):
            job.progress = JobProgress(
                stage="stitching", session_index=i, total_sessions=n, warnings=list(warnings)
            )

            name = session.name
            session_output_dir = output_dir / name
            output_path = session_output_dir / f"{name}_full.mp4"

            stitch_start = time.monotonic()

            def progress_cb(frame: int, total: int, idx: int = i) -> None:
                elapsed = time.monotonic() - stitch_start
                fps = frame / elapsed if elapsed > 0 else 0
                eta = int((total - frame) / fps) if fps > 0 else None
                job.progress = JobProgress(
                    stage="stitching",
                    session_index=idx,
                    total_sessions=n,
                    current_frame=frame,
                    total_frames=total,
                    eta_seconds=eta,
                    warnings=list(warnings),
                )

            try:
                stitch_session(
                    left=Path(session.left_path),
                    right=Path(session.right_path),
                    output=output_path,
                    cal=cal,
                    progress_callback=progress_cb,
                )
            except Exception as e:
                job.progress = JobProgress(
                    stage="failed", session_index=i, total_sessions=n,
                    error=str(e), warnings=list(warnings),
                )
                return

            mobile_path = session_output_dir / f"{name}.mp4"
            transcode_start = time.monotonic()

            def mobile_progress_cb(frame: int, total: int, idx: int = i) -> None:
                elapsed = time.monotonic() - transcode_start
                fps_t = frame / elapsed if elapsed > 0 else 0
                eta = int((total - frame) / fps_t) if fps_t > 0 and total > 0 else None
                job.progress = JobProgress(
                    stage="transcoding",
                    session_index=idx,
                    total_sessions=n,
                    current_frame=frame,
                    total_frames=total,
                    eta_seconds=eta,
                    warnings=list(warnings),
                )

            job.progress = JobProgress(
                stage="transcoding", session_index=i, total_sessions=n, warnings=list(warnings)
            )
            try:
                make_mobile_copy(output_path, mobile_path, mobile_progress_cb)
            except Exception as e:
                job.progress = JobProgress(
                    stage="failed", session_index=i, total_sessions=n,
                    error=str(e), warnings=list(warnings),
                )
                return

            # Outputs with a head edit list play differently per decoder
            # (Firefox shows the pre-trim frames, shifting the timeline) —
            # nothing in this pipeline should produce one.
            from footage_studio.processing.trim import head_edit_list_s
            for out in (output_path, mobile_path):
                elst_s = head_edit_list_s(out)
                if elst_s > 0.2:
                    warnings.append(
                        f"{out.name} carries a {elst_s:.2f}s head edit list — it will "
                        "play out of sync in some browsers. Avoid trimming outputs with "
                        "`ffmpeg -ss ... -c copy`; use footage_studio.processing.trim instead."
                    )

        job.progress = JobProgress(
            stage="done", session_index=n, total_sessions=n, warnings=list(warnings)
        )
