"""Single-command 3D serve analysis with audio contact cue (``analyze-serve``).

Production M4 path for one explicit serve video/range using the
native 3D kinematic chain plus the optional raw-source audio contact cue:

```text
one explicit serve video/range → native-PTS 3D kinematic track
→ segment-safe Butterworth filter (M4.8) → 3D feature matrix (M4.9,
  with aligned raw-audio transient channels when present)
→ six composite anchor candidate sets (M4.10a)
→ six-anchor DP chronology search (M4.10b)
→ two derived midpoint stages → source-frame review
```

Behavior:

- Takes a positional source video plus optional
  ``--start-seconds``/``--end-seconds`` (default: the entire source
  timeline). No ``attempts.json``, ``attempt-id``, phase-feature JSON,
  or other user-supplied JSON is required. A synthetic in-memory
  ``serve-001`` range covering the requested ``[start, end)`` exists
  solely for checkpoint output linkage; M3 detection/cutting is never
  invoked.
- The internal kinematic-track cache reuses the accepted
  ``dense-world-v1`` envelope (exact native PTS, primary 3D joints,
  and quality preserved verbatim). Synchronized normalized 2D
  landmarks are stored only as review companions and are never read
  here: no 2D coordinate feeds filtering, waveforms, candidates, or
  the DP. Legacy sparse ``pose-v1`` caches, ``phase_features.py``,
  ``evidence.py``, and sparse candidates are never invoked on this
  path.
- Orchestrates the existing M4.8 filter, M4.9 waveform, and M4.10a
  composite candidate modules with their default versioned configs,
  then selects the six anchors (``start``, ``release``, ``loading``,
  ``cocking``, ``contact``, ``finish``) with the M4.10b DP that
  preserves the M4.4 recurrence/chronology/transition/skip semantics.
  ``acceleration`` is the exact temporal midpoint of selected
  ``cocking``/``contact`` and ``deceleration`` the exact midpoint of
  selected ``contact``/``finish``; a derived stage is unavailable when
  either bounding anchor is unavailable.
- This path demuxes raw source audio once with the existing
  ``media.audio`` API, aligns it to the exact kinematic PTS grid, and
  passes it to the M4.9 waveform builder as the optional aligned audio
  sequence. Demux/alignment failure or an absent audio stream is
  nonfatal and yields honestly unavailable audio channels. ``contact``
  uses ``body_pose_audio`` provenance iff the selected contact carries
  a strictly positive/true supporting ``audio_transient`` cue value
  (``> 0``), otherwise ``body_pose``; a merely available ``0.0`` flag
  never qualifies.
- Writes a normal compatible ``checkpoints.json``
  (:class:`PhaseDocument`), a compact deterministic 3D diagnostic
  JSON, and a review directory with source-frame JPEGs plus a
  self-contained ``index.html`` rendered directly from the selected
  timestamps. Review JPEGs are clean source frames; a skeleton overlay
  is not required and never substitutes for evidence.
- Atomic output/cache behavior: ``checkpoints.json`` and the
  diagnostic JSON are written to temporary siblings and atomically
  renamed; the review directory is staged in a temporary sibling and
  atomically renamed; the kinematic cache uses the atomic
  dense-world writers (hit/resume/quarantine/collision semantics).
  Failure or cancellation writes no partial final file and removes
  temporary siblings. Unavailable stages honestly produce no image
  but appear in the review index.

All subprocesses use argument arrays via the underlying media/pose
adapters. Canonical source time comes from probe metadata and exact
native PTS, never from a frame index or assumed FPS.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from serve_review.checkpoints import composite_anchors as composite_module
from serve_review.checkpoints import kinematic_waveforms as waveforms_module
from serve_review.checkpoints import scene_features as scene_features_module
from serve_review.checkpoints import six_anchor_solver as six_anchor_module
from serve_review.checkpoints import world_filter as filter_module
from serve_review.checkpoints.composite_anchors import (
    COMPOSITE_ANCHOR_STAGES,
    CompositeAnchorCandidate,
    CompositeAnchorConfig,
    CompositeAnchorSet,
)
from serve_review.checkpoints.kinematic_waveforms import KinematicWaveformsConfig
from serve_review.checkpoints.six_anchor_solver import (
    SIX_ANCHOR_STAGES,
    SixAnchorSolverConfig,
)
from serve_review.checkpoints.world_filter import WorldFilterConfig
from serve_review.detection.decoder import DecoderConfig
from serve_review.domain import (
    STAGE_ORDER,
    AttemptPhase,
    MediaRange,
    PhaseDocument,
    SourceMetadata,
    StagePhase,
)
from serve_review.fingerprint import ARTIFACT_FILENAME as FINGERPRINT_FILENAME
from serve_review.fingerprint import build_serve_fingerprint_v1
from serve_review.media import audio as audio_module
from serve_review.media import color as color_module
from serve_review.media import frames as frames_module
from serve_review.media import probe as probe_module
from serve_review.pose import mediapipe as mediapipe_module
from serve_review.pose import world as world_module
from serve_review.pose.world import WorldFrameObservation
from serve_review.scene import build_scene_track_with_racketvision_observations
from serve_review.scene_diagnostics import build_scene_visual_diagnostics
from serve_review.tracking.cache import (
    RacketVisionCacheCorruptError,
    RacketVisionCacheError,
    RacketVisionCacheIdentity,
    RacketVisionCacheStaleError,
    fingerprint_frame_times,
    load_racketvision_cache,
    write_racketvision_cache,
)
from serve_review.tracking.model_frames import iter_racketvision_model_frames
from serve_review.tracking.racketvision import (
    RacketVisionConfig,
    RacketVisionTracker,
    fingerprint_racketvision_config,
)

__all__ = [
    "ANALYZE_SERVE_VERSION",
    "ANALYZE_SERVE_METHOD_VERSION",
    "ANALYZE_SERVE_SCHEMA_VERSION",
    "ANALYZE_SERVE_ATTEMPT_ID",
    "ANALYZE_SERVE_ANCHOR_STAGES",
    "ANCHOR2_VIDEO_BASENAME",
    "ANCHOR2_ANNOTATION_RELPATH",
    "CHECKPOINTS_FILENAME",
    "DIAGNOSTICS_FILENAME",
    "REVIEW_DIRNAME",
    "REVIEW_JSON_FILENAME",
    "INDEX_HTML_FILENAME",
    "KINEMATIC_CACHE_FILENAME",
    "RACKETVISION_CACHE_FILENAME",
    "CACHE_FLUSH_INTERVAL_FRAMES",
    "AnalyzeServeError",
    "AnalyzeServeCancelled",
    "AnalyzeServeResult",
    "default_kinematic_cache_path_for",
    "default_racketvision_cache_path_for",
    "run_analyze_serve",
]

#: Orchestration identity recorded in diagnostics/review manifests.
ANALYZE_SERVE_VERSION = "analyze-serve-v1"
#: Method identity recorded on the emitted attempt phase.
ANALYZE_SERVE_METHOD_VERSION = "serve-waveform-v1"
#: Schema version of the 3D diagnostic JSON.
ANALYZE_SERVE_SCHEMA_VERSION = 1
#: Synthetic in-memory attempt id used solely for output linkage.
ANALYZE_SERVE_ATTEMPT_ID = "serve-001"
#: Anchor stages searched by the DP (derived stages follow afterwards).
ANALYZE_SERVE_ANCHOR_STAGES: tuple[str, ...] = SIX_ANCHOR_STAGES
#: M4 production calibration: finish must occur within this bound of contact.
ANALYZE_SERVE_MAX_CONTACT_TO_FINISH_SECONDS = 0.80
#: Distinguishes the M4-specific constrained solver from the shared default.
ANALYZE_SERVE_SOLVER_CONFIG_ID = "phase-solver-serve-default-v2"

#: Fixed basename for the one known anchor video supporting --anchor2comparison.
ANCHOR2_VIDEO_BASENAME = "single-serve-02.mov"
#: Fixed minimal manual-anchor JSON loaded only when --anchor2comparison is set.
ANCHOR2_ANNOTATION_RELPATH = Path(
    "refs/annotations/dev/single-serve-02.anchor2.json"
)

#: Final checkpoint filename written atomically beside review outputs.
CHECKPOINTS_FILENAME = "checkpoints.json"
#: Compact deterministic 3D diagnostic filename.
DIAGNOSTICS_FILENAME = "serve-3d-diagnostics.json"
#: Human review directory name under the session directory.
REVIEW_DIRNAME = "review-serve-3d"
#: Machine-readable deterministic review index filename.
REVIEW_JSON_FILENAME = "review.json"
#: Human-readable self-contained review page filename.
INDEX_HTML_FILENAME = "index.html"
#: Default kinematic-track (dense-world envelope) cache filename.
KINEMATIC_CACHE_FILENAME = "kinematic-track-v1.jsonl"
#: Default raw RacketVision observation cache filename.
RACKETVISION_CACHE_FILENAME = "racketvision-track-v1.jsonl"
#: How often to atomically flush a resumable partial kinematic cache.
CACHE_FLUSH_INTERVAL_FRAMES = 30

#: Tolerance for requested-duration versus probed-duration agreement.
_DURATION_TOLERANCE_SECONDS = 1e-6
#: Half-open linkage interval width for point estimates (ordering-safe).
_POINT_INTERVAL_EPSILON_SECONDS = 1e-6


class AnalyzeServeError(Exception):
    """Actionable analyze-serve failure at one stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.message}"


class AnalyzeServeCancelled(AnalyzeServeError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


@dataclass(frozen=True, slots=True)
class AnalyzeServeResult:
    """Outcome of :func:`run_analyze_serve`."""

    video: Path
    session_dir: Path
    source_metadata: SourceMetadata
    requested_range: MediaRange
    attempt_range: MediaRange
    attempt_phase: AttemptPhase
    checkpoints_path: Path
    diagnostics_path: Path
    fingerprint_path: Path
    review_dir: Path
    index_html: Path
    review_json: Path
    cache_path: Path
    cache_hit: bool
    racketvision_cache_path: Path
    racketvision_cache_hit: bool
    frame_count: int
    inferred_frames: int
    total_score: float
    selected_times: dict | None = None
    image_count: int = 0
    entry_count: int = 0
    empty: bool = False


def default_kinematic_cache_path_for(
    video: Path | str, output_dir: Path | str = "output"
) -> Path:
    """Return the default reusable kinematic-track cache path for ``video``.

    The file lives beneath the exact output directory:
    ``<output_dir>/cache/kinematic-track-v1.jsonl``. The
    file uses the accepted ``dense-world-v1`` envelope so repeated runs
    over the same range reuse inference-free rows.
    """
    stem = Path(str(video)).stem
    if not stem:
        raise AnalyzeServeError(
            "validate", f"invalid video path {video!r}: no file stem."
        )
    return (
        Path(output_dir).expanduser()
        / "cache"
        / KINEMATIC_CACHE_FILENAME
    )


def default_racketvision_cache_path_for(
    video: Path | str, output_dir: Path | str = "output"
) -> Path:
    """Return the default reusable raw object-track cache path."""
    stem = Path(str(video)).stem
    if not stem:
        raise AnalyzeServeError(
            "validate", f"invalid video path {video!r}: no file stem."
        )
    return Path(output_dir).expanduser() / "cache" / RACKETVISION_CACHE_FILENAME


def _check_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalyzeServeError(
            "validate", f"invalid {name}: {value!r}; expected True or False."
        )
    return value


def _check_executable(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalyzeServeError(
            "validate", f"invalid {name}: {value!r}; expected a non-blank name."
        )
    return value.strip()


def _check_optional_bound(
    label: str, value: object, *, allow_none: bool = True
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalyzeServeError(
            "validate",
            f"invalid --{label} {value!r}; expected seconds as a number.",
        )
    number = float(value)
    if not math.isfinite(number):
        raise AnalyzeServeError(
            "validate",
            f"invalid --{label} {value!r}; expected a finite number of seconds.",
        )
    return number


def _resolve_range(
    start_raw: float | None,
    end_raw: float | None,
    duration: float,
) -> tuple[float, float]:
    start = 0.0 if start_raw is None else float(start_raw)
    end = float(duration) if end_raw is None else float(end_raw)
    if start < 0:
        raise AnalyzeServeError(
            "validate",
            f"invalid --start-seconds {start_raw!r}; expected seconds >= 0.",
        )
    if end < 0:
        raise AnalyzeServeError(
            "validate",
            f"invalid --end-seconds {end_raw!r}; expected seconds >= 0.",
        )
    if not end > start:
        raise AnalyzeServeError(
            "validate",
            f"invalid serve range [{start!r}, {end!r}); "
            "expected --start-seconds < --end-seconds.",
        )
    if start >= duration:
        raise AnalyzeServeError(
            "validate",
            f"serve range start {start!r} lies outside the source timeline "
            f"[0, {duration}); expected 0 <= start < duration.",
        )
    if end > duration + frames_module.FRAME_MATCH_TOLERANCE_SECONDS:
        raise AnalyzeServeError(
            "validate",
            f"serve range end {end!r} lies beyond the source duration "
            f"({duration!r}); expected end <= duration.",
        )
    return start, min(end, duration)


def _write_text_atomic(target: Path, text: str, stage: str) -> Path:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AnalyzeServeError(
                stage, f"could not create output directory {parent}: {exc}."
            ) from exc
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=target.name + ".tmp-",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except AnalyzeServeError:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise AnalyzeServeError(
            stage, f"could not write output file {target}: {exc}."
        ) from exc
    return target


def _remove_tmp_siblings(directory: Path, stem: str) -> None:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(stem + ".tmp-"):
            try:
                entry.unlink(missing_ok=True)
            except OSError:
                pass


def _quarantine_quietly(path: Path) -> None:
    try:
        world_module.quarantine_world_cache(path)
    except world_module.CacheError:
        pass


def _encode_jpeg_ffmpeg(image: np.ndarray, dest: Path, ffmpeg: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise AnalyzeServeError(
            "review", "invalid annotated image for JPEG encoding."
        )
    height, width, _ = image.shape
    if height <= 0 or width <= 0:
        raise AnalyzeServeError(
            "review", "invalid annotated image dimensions."
        )
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(dest),
    ]
    try:
        completed = subprocess.run(
            args,
            input=np.ascontiguousarray(image, dtype=np.uint8).tobytes(),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AnalyzeServeError(
            "review",
            f"ffmpeg was not found as {ffmpeg!r}; install FFmpeg and "
            "ensure it is on PATH.",
        ) from exc
    except OSError as exc:
        raise AnalyzeServeError(
            "review", f"could not run ffmpeg as {ffmpeg!r}: {exc}."
        ) from exc
    if completed.returncode != 0:
        detail = (
            (completed.stderr or b"").decode("utf-8", "replace").strip()
            if isinstance(completed.stderr, bytes)
            else str(completed.stderr or "").strip()
        )
        suffix = f": {detail}" if detail else ""
        raise AnalyzeServeError(
            "review",
            f"ffmpeg failed encoding JPEG for {dest.name} "
            f"(exit {completed.returncode}){suffix}.",
        )
    try:
        if not dest.is_file() or dest.stat().st_size == 0:
            raise AnalyzeServeError(
                "review", f"ffmpeg did not produce JPEG output at {dest}."
            )
    except OSError as exc:
        raise AnalyzeServeError(
            "review", f"could not verify JPEG output at {dest}: {exc}."
        ) from exc


def _load_anchor2_manual() -> dict[str, float | None]:
    """Load the fixed minimal anchor2 manual PTS mapping.

    Reads only ``ANCHOR2_ANNOTATION_RELPATH`` (a flat stage-to-seconds
    object with no schema/version/fingerprint metadata). Missing stages
    read as None; present values must be finite numbers.
    """
    path = ANCHOR2_ANNOTATION_RELPATH
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalyzeServeError(
            "validate",
            f"could not read anchor2 manual file {path}: {exc}.",
        ) from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AnalyzeServeError(
            "validate",
            f"invalid anchor2 manual JSON in {path}: {exc}.",
        ) from exc
    if not isinstance(payload, dict):
        raise AnalyzeServeError(
            "validate",
            f"invalid anchor2 manual in {path}: expected an object of "
            "stage to source seconds.",
        )
    manual: dict[str, float | None] = {}
    for stage in STAGE_ORDER:
        value = payload.get(stage)
        if value is None:
            manual[stage] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalyzeServeError(
                "validate",
                f"invalid anchor2 manual stage {stage!r}: {value!r}; "
                "expected source seconds as a number.",
            )
        number = float(value)
        if not math.isfinite(number):
            raise AnalyzeServeError(
                "validate",
                f"invalid anchor2 manual stage {stage!r}: {value!r}; "
                "expected a finite number of seconds.",
            )
        manual[stage] = number
    return manual


def _safe_component(name: str) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name
    )
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def _build_index_html(
    entries: list[dict[str, Any]],
    *,
    fingerprint: str,
    attempt_id: str,
    range_text: str,
    anchor2comparison: bool = False,
) -> str:
    rows: list[str] = []
    for entry in entries:
        img = entry.get("image")
        if img is None:
            cell = "<em>no image</em>"
        else:
            cell = (
                f'<a href="{img}"><img src="{img}" alt="{entry.get("stage")}" '
                'style="max-width:320px"></a>'
            )
        if anchor2comparison:
            manual_img = entry.get("manual_image")
            if manual_img is None:
                manual_cell = "<em>no image</em>"
            else:
                manual_cell = (
                    f'<a href="{manual_img}"><img src="{manual_img}" '
                    f'alt="manual-{entry.get("stage")}" '
                    'style="max-width:320px"></a>'
                )
            rows.append(
                "<tr>"
                f"<td>{entry.get('stage')}</td>"
                f"<td>{entry.get('availability')}</td>"
                f"<td>{entry.get('provenance')}</td>"
                f"<td>{entry.get('confidence'):.2f}</td>"
                f"<td>{entry.get('requested_source_time')}</td>"
                f"<td>{entry.get('actual_source_time')}</td>"
                f"<td>{cell}</td>"
                f"<td>{entry.get('manual_requested_source_time')}</td>"
                f"<td>{entry.get('manual_actual_source_time')}</td>"
                f"<td>{entry.get('delta_selected_minus_manual_ms')}</td>"
                f"<td>{manual_cell}</td>"
                f"<td>{entry.get('status')}</td>"
                "</tr>"
            )
            continue
        rows.append(
            "<tr>"
            f"<td>{entry.get('stage')}</td>"
            f"<td>{entry.get('availability')}</td>"
            f"<td>{entry.get('provenance')}</td>"
            f"<td>{entry.get('confidence'):.2f}</td>"
            f"<td>{entry.get('requested_source_time')}</td>"
            f"<td>{entry.get('actual_source_time')}</td>"
            f"<td>{cell}</td>"
            f"<td>{entry.get('status')}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    if anchor2comparison:
        header = (
            "<tr><th>stage</th><th>availability</th><th>provenance</th>"
            "<th>confidence</th><th>requested s</th><th>frame s</th>"
            "<th>image</th><th>manual requested s</th>"
            "<th>manual frame s</th><th>delta selected-minus-manual ms</th>"
            "<th>manual image</th><th>status</th></tr>\n"
        )
    else:
        header = (
            "<tr><th>stage</th><th>availability</th><th>provenance</th>"
            "<th>confidence</th><th>requested s</th><th>frame s</th>"
            "<th>image</th><th>status</th></tr>\n"
        )
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>Serve review {attempt_id}</title></head><body>\n"
        f"<h1>Serve review {attempt_id} ({fingerprint})</h1>\n"
        f"<p>Serve range (source seconds): {range_text}. Timestamps are "
        "canonical source times; labels and frame times are recorded in "
        "this index and review.json.</p>\n"
        "<table border=\"1\">\n"
        f"{header}"
        f"{body}\n</table>\n</body></html>\n"
    )


def _attach_shared_transient_flags(
    aligned: Sequence[Any],
    frame_times: Sequence[float],
) -> list[Any]:
    """Attach shared-policy explicit 1/0 flags to aligned audio rows.

    Converts each aligned row to an ``(energy, flag)`` pair whose flag
    comes from the single shared
    :func:`serve_review.media.audio.qualify_audio_transients` policy
    with ``DecoderConfig`` defaults (``1.0`` at qualified transient
    times, ``0.0`` elsewhere), so the waveform builder never falls back
    to strict local energy maxima. Rows without energy stay ``None``
    (honestly unavailable). No threshold, baseline, or peak-picking
    logic lives here: energy extraction only, qualification delegated.
    """
    defaults = DecoderConfig()
    rows = list(aligned)
    times = [float(moment) for moment in frame_times]
    if len(rows) != len(times):
        raise AnalyzeServeError(
            "solve",
            "aligned audio holds "
            f"{len(rows)} row(s) for {len(times)} frame time(s); "
            "audio must align one-for-one in order.",
        )

    def _row_energy(entry: Any) -> float | None:
        if entry is None:
            return None
        if isinstance(entry, bool):
            raise AnalyzeServeError(
                "solve", f"invalid aligned audio energy: {entry!r}."
            )
        if isinstance(entry, (int, float)):
            number = float(entry)
            if not math.isfinite(number) or number < 0.0:
                raise AnalyzeServeError(
                    "solve",
                    f"invalid aligned audio energy: {entry!r}.",
                )
            return number
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            raw = entry[0]
            if raw is None:
                return None
            return _row_energy(raw)
        if isinstance(entry, dict):
            raw_energy: Any = None
            found = False
            for key in ("energy", "audio_energy", "value"):
                if key in entry:
                    raw_energy = entry[key]
                    found = True
                    break
            if not found:
                raise AnalyzeServeError(
                    "solve",
                    "aligned audio mappings must carry an 'energy' key.",
                )
            if raw_energy is None:
                return None
            return _row_energy(raw_energy)
        energy_attr = getattr(entry, "energy", None)
        if energy_attr is not None:
            return _row_energy(energy_attr)
        return _row_energy(float(entry))  # type: ignore[arg-type]

    energies = [_row_energy(entry) for entry in rows]
    present = tuple(
        audio_module.AudioEnergy(time_seconds=moment, energy=energy)
        for moment, energy in zip(times, energies)
        if energy is not None
    )
    try:
        qualified = set(
            audio_module.qualify_audio_transients(
                present,
                float(defaults.audio_transient_ratio),
                float(defaults.audio_transient_floor),
            )
        )
    except audio_module.AudioError as exc:
        raise AnalyzeServeError(
            "solve", f"shared transient qualification failed: {exc}."
        ) from exc
    explicit: list[Any] = []
    for moment, energy in zip(times, energies):
        if energy is None:
            explicit.append(None)
        else:
            explicit.append(
                (float(energy), 1.0 if float(moment) in qualified else 0.0)
            )
    return explicit


def _scene_audio_rows(
    aligned: Sequence[Any] | None,
    frame_times: Sequence[float],
) -> tuple[tuple[audio_module.AudioEnergy | None, ...] | None, tuple[bool | None, ...] | None]:
    """Project qualified aligned audio into SceneTrack row observations."""
    if aligned is None:
        return (None, None)
    if len(aligned) != len(frame_times):
        raise AnalyzeServeError(
            "solve", "qualified audio must contain one row per scene PTS."
        )
    energies: list[audio_module.AudioEnergy | None] = []
    transients: list[bool | None] = []
    for moment, row in zip(frame_times, aligned):
        if row is None:
            energies.append(None)
            transients.append(None)
            continue
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise AnalyzeServeError("solve", "invalid qualified audio row.")
        energy, flag = row
        if energy is None:
            energies.append(None)
            transients.append(None)
            continue
        energies.append(audio_module.AudioEnergy(float(moment), float(energy)))
        transients.append(bool(flag) if flag is not None else None)
    return (tuple(energies), tuple(transients))


def _structural_status(availabilities: Sequence[str]) -> str:
    available = sum(1 for value in availabilities if value == "available")
    unavailable = sum(1 for value in availabilities if value == "unavailable")
    total = len(availabilities)
    if available == total:
        return "complete"
    if unavailable == total:
        return "unavailable"
    if available >= 1:
        return "partial"
    return "incomplete"


def run_analyze_serve(
    video: Path | str,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    output_dir: Path | str | None = None,
    cache_path: Path | str | None = None,
    racketvision_cache_path: Path | str | None = None,
    racketvision_config: RacketVisionConfig | None = None,
    racketvision_tracker_fingerprint: str | None = None,
    racketvision_frame_factory: Callable[..., Iterator[Any]] | None = None,
    racketvision_tracker_factory: Callable[[RacketVisionConfig], Any] | None = None,
    model_path: Path | str = mediapipe_module.DEFAULT_MODEL_PATH,
    dry_run: bool = False,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    filter_config: WorldFilterConfig | None = None,
    waveforms_config: KinematicWaveformsConfig | None = None,
    composite_config: CompositeAnchorConfig | None = None,
    solver_config: SixAnchorSolverConfig | None = None,
    probe_fn: Callable[[Path], SourceMetadata] | None = None,
    native_times_fn: Callable[..., tuple[float, ...]] | None = None,
    native_frame_factory: Callable[..., Iterator[Any]] | None = None,
    backend_factory: Callable[[], Any] | None = None,
    sample_frame_fn: Callable[..., Any] | None = None,
    encode_jpeg_fn: Callable[[np.ndarray, Path], None] | None = None,
    audio_energies_fn: Callable[..., Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    anchor2comparison: bool = False,
) -> AnalyzeServeResult | None:
    """Analyze one explicit serve range through the 3D path with audio cue.

    Args:
        video: Source video path; read only, never modified.
        start_seconds/end_seconds: Half-open ``[start, end)`` serve
            range in source seconds (default: the entire source).
        output_dir: Exact output directory. Defaults to
            ``VIDEO.parent/metadata/VIDEO.stem/manual-analysis``; an
            explicit value is used exactly with no stem appended.
        cache_path: Explicit reusable kinematic-track cache file
            (dense-world envelope); defaults to
            ``<output_dir>/cache/kinematic-track-v1.jsonl``.
        racketvision_cache_path: Explicit reusable raw object-track cache;
            defaults to ``<output_dir>/cache/racketvision-track-v1.jsonl``.
            RacketVision runs automatically on a cache miss.
        model_path: Approved Pose Landmarker ``.task`` artifact.
        dry_run: Validate the request and report intended destinations
            without directory creation, cache/model work, media export,
            or artifact writes; returns None.
        force: Replace only this command's exact output destination
            when True; refuse on collision when False.
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        filter_config/waveforms_config/composite_config/solver_config:
            Versioned M4 configs (defaults are the modules' defaults;
            weights are never tuned here).
        probe_fn/native_times_fn/native_frame_factory/backend_factory/
        sample_frame_fn/encode_jpeg_fn/audio_energies_fn: Injectable stage
            functions for tests. When None, the real media/pose/audio
            adapters run. ``audio_energies_fn`` receives
            ``(video_path, frame_times)`` (exact kinematic PTS tuple) and
            returns an aligned audio sequence with exactly one entry per
            frame time, or ``None`` for unavailable audio.
        progress_callback: Optional ``(stage_message)`` hook.
        is_cancelled: Optional hook; a true return raises
            :class:`AnalyzeServeCancelled` and writes no final file.

    Returns:
        An :class:`AnalyzeServeResult`, or None when ``dry_run`` is True.

    Raises:
        AnalyzeServeError: Stage-specific failure (``.stage`` names
            the stage: ``validate``, ``probe``, ``kinematics``,
            ``solve``, ``review``, or ``write``).
        AnalyzeServeCancelled: On cancellation; no final file written.
    """
    started = time.monotonic()

    def _progress(message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                pass

    def _cancelled() -> bool:
        if is_cancelled is None:
            return False
        if not callable(is_cancelled):
            raise AnalyzeServeError(
                "validate", f"invalid is_cancelled: {is_cancelled!r}."
            )
        return bool(is_cancelled())

    def _fail(stage: str, message: str) -> AnalyzeServeError:
        return AnalyzeServeError(stage, message)

    video_path = Path(video).expanduser()
    if not str(video_path):
        raise _fail("validate", "invalid video: expected a non-blank path.")
    if not video_path.is_file():
        raise _fail("validate", f"input video does not exist: {video_path}.")
    dry_run_flag = _check_bool(dry_run, "dry_run")
    force_flag = _check_bool(force, "force")
    anchor2_flag = _check_bool(anchor2comparison, "anchor2comparison")
    if anchor2_flag and video_path.name != ANCHOR2_VIDEO_BASENAME:
        raise _fail(
            "validate",
            f"--anchor2comparison supports only {ANCHOR2_VIDEO_BASENAME}; "
            f"got {video_path.name!r}.",
        )
    manual_by_stage: dict[str, float | None] | None = None
    if anchor2_flag:
        manual_by_stage = _load_anchor2_manual()
    ffmpeg_exe = _check_executable("ffmpeg", ffmpeg)
    ffprobe_exe = _check_executable("ffprobe", ffprobe)
    start_opt = _check_optional_bound("start-seconds", start_seconds)
    end_opt = _check_optional_bound("end-seconds", end_seconds)
    if start_opt is not None and start_opt < 0:
        raise _fail(
            "validate",
            f"invalid --start-seconds {start_seconds!r}; "
            "expected seconds >= 0.",
        )
    if end_opt is not None and end_opt < 0:
        raise _fail(
            "validate",
            f"invalid --end-seconds {end_seconds!r}; expected seconds >= 0.",
        )
    if racketvision_tracker_fingerprint is not None and (
        not isinstance(racketvision_tracker_fingerprint, str)
        or not racketvision_tracker_fingerprint.strip()
    ):
        raise _fail(
            "validate", "invalid racketvision_tracker_fingerprint: expected text."
        )
    for label, fn in (
        ("probe_fn", probe_fn),
        ("native_times_fn", native_times_fn),
        ("native_frame_factory", native_frame_factory),
        ("backend_factory", backend_factory),
        ("sample_frame_fn", sample_frame_fn),
        ("encode_jpeg_fn", encode_jpeg_fn),
        ("audio_energies_fn", audio_energies_fn),
        ("racketvision_frame_factory", racketvision_frame_factory),
        ("racketvision_tracker_factory", racketvision_tracker_factory),
    ):
        if fn is not None and not callable(fn):
            raise _fail("validate", f"invalid {label}: {fn!r}.")
    cfg_filter = (
        filter_config if filter_config is not None else WorldFilterConfig()
    )
    if not isinstance(cfg_filter, WorldFilterConfig):
        raise _fail(
            "validate",
            "invalid filter_config: expected WorldFilterConfig, "
            f"got {type(cfg_filter).__name__}.",
        )
    cfg_waveforms = (
        waveforms_config
        if waveforms_config is not None
        else KinematicWaveformsConfig()
    )
    if not isinstance(cfg_waveforms, KinematicWaveformsConfig):
        raise _fail(
            "validate",
            "invalid waveforms_config: expected KinematicWaveformsConfig, "
            f"got {type(cfg_waveforms).__name__}.",
        )
    cfg_composite = (
        composite_config
        if composite_config is not None
        else CompositeAnchorConfig()
    )
    if not isinstance(cfg_composite, CompositeAnchorConfig):
        raise _fail(
            "validate",
            "invalid composite_config: expected CompositeAnchorConfig, "
            f"got {type(cfg_composite).__name__}.",
        )
    cfg_solver = (
        solver_config
        if solver_config is not None
        else SixAnchorSolverConfig(
            config_id=ANALYZE_SERVE_SOLVER_CONFIG_ID,
            max_contact_to_finish_seconds=ANALYZE_SERVE_MAX_CONTACT_TO_FINISH_SECONDS,
        )
    )
    if not isinstance(cfg_solver, SixAnchorSolverConfig):
        raise _fail(
            "validate",
            "invalid solver_config: expected SixAnchorSolverConfig, "
            f"got {type(cfg_solver).__name__}.",
        )
    if isinstance(model_path, Path):
        model_target: Path | str = model_path.expanduser()
    elif isinstance(model_path, str):
        if not model_path.strip():
            raise _fail(
                "validate", "invalid model path: expected a non-blank path."
            )
        model_target = Path(model_path.strip()).expanduser()
    else:
        raise _fail(
            "validate",
            f"invalid model path: {model_path!r}; expected a path.",
        )

    if output_dir is None:
        session_dir = (
            video_path.parent / "metadata" / video_path.stem / "manual-analysis"
        )
    else:
        session_dir = Path(output_dir).expanduser()
        if not str(session_dir):
            raise _fail("validate", "invalid output directory: expected a non-blank path.")
    checkpoints_out = session_dir / CHECKPOINTS_FILENAME
    diagnostics_out = session_dir / DIAGNOSTICS_FILENAME
    fingerprint_out = session_dir / FINGERPRINT_FILENAME
    review_dir = session_dir / REVIEW_DIRNAME
    if cache_path is not None:
        cache_target = Path(cache_path).expanduser()
    else:
        cache_target = session_dir / "cache" / KINEMATIC_CACHE_FILENAME
    if not str(cache_target):
        raise _fail("validate", "invalid cache path: expected a non-blank path.")
    racketvision_cache_target = (
        Path(racketvision_cache_path).expanduser()
        if racketvision_cache_path is not None
        else session_dir / "cache" / RACKETVISION_CACHE_FILENAME
    )
    if racketvision_cache_target.exists() and racketvision_cache_target.is_dir():
        raise _fail(
            "validate",
            f"invalid RacketVision cache path {racketvision_cache_target}: "
            "destination is a directory.",
        )
    if cache_target.exists() and cache_target.is_dir():
        raise _fail(
            "validate",
            f"invalid cache path {cache_target}: destination is a directory.",
        )
    try:
        resolved_cache = cache_target.resolve()
        try:
            if video_path.resolve() == resolved_cache:
                raise _fail(
                    "validate",
                    f"cache path collides with the source video {video_path}; "
                    "choose a distinct --cache path (the source is never "
                    "modified).",
                )
        except OSError:
            pass
    except AnalyzeServeError:
        raise

    if dry_run_flag:
        _progress(f"analyze-serve dry-run: video {video_path}")
        _progress(f"analyze-serve dry-run: output {session_dir}")
        _progress(f"analyze-serve dry-run: cache {cache_target}")
        _progress(
            f"analyze-serve dry-run: RacketVision cache "
            f"{racketvision_cache_target}"
        )
        if start_opt is not None or end_opt is not None:
            _progress(
                f"analyze-serve dry-run: range {start_opt} {end_opt}"
            )
        return None

    try:
        # --- probe ---
        _progress(f"analyze-serve: probing {video_path}")
        if _cancelled():
            raise AnalyzeServeCancelled(
                "probe", "analyze-serve was cancelled before probing."
            )
        resolve_probe = (
            probe_fn
            if probe_fn is not None
            else (lambda path: probe_module.probe_source(path, ffprobe=ffprobe_exe))
        )
        try:
            metadata = resolve_probe(video_path)
        except AnalyzeServeCancelled:
            raise
        except AnalyzeServeError:
            raise
        except Exception as exc:
            raise _fail("probe", f"probe failed for {video_path}: {exc}.") from exc
        if not isinstance(metadata, SourceMetadata):
            raise _fail(
                "probe",
                f"probe returned {type(metadata).__name__}; "
                "expected SourceMetadata.",
            )
        try:
            duration = float(metadata.duration_seconds)
            fingerprint = str(metadata.fingerprint)
        except (AttributeError, TypeError, ValueError) as exc:
            raise _fail(
                "probe", f"invalid source metadata for {video_path}: {exc}."
            ) from exc
        if not fingerprint.strip():
            raise _fail(
                "probe",
                f"invalid source metadata for {video_path}: blank fingerprint.",
            )
        if not math.isfinite(duration) or duration <= 0:
            raise _fail(
                "probe",
                f"invalid source metadata for {video_path}: "
                f"bad duration {duration!r}.",
            )
        try:
            color_filter = color_module.probe_color_metadata(
                video_path, ffprobe=ffprobe_exe
            ).sdr_filter
        except frames_module.FrameError as exc:
            raise _fail("probe", f"could not inspect source color metadata: {exc}.") from exc

        # --- requested range (default: entire source) ---
        req_start, req_end = _resolve_range(start_opt, end_opt, duration)
        try:
            requested_range = MediaRange(req_start, req_end)
        except Exception as exc:
            raise _fail("validate", f"invalid serve range: {exc}.") from exc

        # --- native PTS grid + kinematic-track cache ---
        _progress("analyze-serve: ensuring kinematic track")
        if _cancelled():
            raise AnalyzeServeCancelled(
                "kinematics",
                "analyze-serve was cancelled before kinematic extraction.",
            )
        try:
            if native_times_fn is not None:
                expected = tuple(native_times_fn(video_path, req_start, req_end))
            else:
                expected = frames_module.native_frame_times(
                    video_path,
                    req_start,
                    req_end,
                    source=metadata,
                    ffprobe=ffprobe_exe,
                )
        except AnalyzeServeCancelled:
            raise
        except AnalyzeServeError:
            raise
        except frames_module.FrameError as exc:
            raise _fail("kinematics", f"could not list native frames: {exc}.") from exc
        except Exception as exc:
            raise _fail("kinematics", f"could not list native frames: {exc}.") from exc
        if not expected:
            raise _fail(
                "kinematics",
                f"serve range [{req_start!r}, {req_end!r}) holds no decodable "
                "source frame.",
            )
        for moment in expected:
            if (
                isinstance(moment, bool)
                or not isinstance(moment, (int, float))
                or not math.isfinite(float(moment))
            ):
                raise _fail(
                    "kinematics",
                    f"native frame grid holds an invalid time {moment!r}.",
                )
        expected = tuple(float(moment) for moment in expected)
        for earlier, later in zip(expected, expected[1:]):
            if not later > earlier:
                raise _fail(
                    "kinematics",
                    "native frame grid must be strictly increasing; refusing "
                    "to emit misaligned kinematic rows.",
                )
        total = len(expected)

        if backend_factory is not None:
            try:
                backend = backend_factory()
            except (AnalyzeServeError, AnalyzeServeCancelled):
                raise
            except Exception as exc:
                raise _fail(
                    "kinematics", f"could not open pose backend: {exc}."
                ) from exc
        else:
            try:
                backend = mediapipe_module.open_mediapipe_backend(
                    model_target, num_poses=1
                )
            except Exception as exc:
                raise _fail("kinematics", f"{exc}") from exc
        try:
            infer_world = getattr(backend, "infer_world", None)
            if not callable(infer_world):
                raise _fail(
                    "kinematics",
                    "pose backend exposes no callable 'infer_world'; "
                    "refusing to substitute sparse 2D inference for "
                    "kinematic-track rows.",
                )
            try:
                model_name = str(backend.model_name)
                model_version = str(backend.model_version)
            except Exception as exc:
                raise _fail(
                    "kinematics",
                    f"pose backend exposes invalid model identity: {exc}.",
                ) from exc
            if not model_name.strip() or not model_version.strip():
                raise _fail(
                    "kinematics",
                    "pose backend exposes a blank model name/version; "
                    "refusing to write an unidentified cache.",
                )
            try:
                identity = world_module.WorldCacheIdentity(
                    source_fingerprint=fingerprint,
                    model_name=model_name,
                    model_version=(
                        f"{model_version}+input-{color_module.MODEL_INPUT_PREPROCESS_VERSION}"
                    ),
                )
            except Exception as exc:
                raise _fail(
                    "kinematics", f"invalid cache identity: {exc}."
                ) from exc

            cached: list[WorldFrameObservation] = []
            cache_hit = False
            if cache_target.is_file() and not force_flag:
                try:
                    snapshot = world_module.load_world_cache(cache_target)
                except world_module.CacheStaleError:
                    _quarantine_quietly(cache_target)
                    snapshot = None
                except world_module.CacheCorruptError:
                    _quarantine_quietly(cache_target)
                    snapshot = None
                except world_module.CacheError as exc:
                    raise _fail(
                        "kinematics",
                        f"could not read kinematic-track cache "
                        f"{cache_target}: {exc}.",
                    ) from exc
                if snapshot is not None:
                    try:
                        world_module.require_matching_world_identity(
                            snapshot.identity, identity
                        )
                    except world_module.CacheStaleError:
                        _quarantine_quietly(cache_target)
                        snapshot = None
                    else:
                        stored_times = tuple(
                            frame.time_seconds for frame in snapshot.frames
                        )
                        if snapshot.complete:
                            if tuple(stored_times) == tuple(expected):
                                cached = list(snapshot.frames)
                                cache_hit = True
                            else:
                                raise _fail(
                                    "kinematics",
                                    f"kinematic-track cache {cache_target} is "
                                    "complete but does not exactly match the "
                                    f"{total} native frame(s) in "
                                    f"[{req_start!r}, {req_end!r}); pass "
                                    "--force to replace it or choose a "
                                    "distinct --cache path.",
                                )
                        else:
                            prefix = tuple(expected[: len(stored_times)])
                            if tuple(stored_times) != prefix:
                                raise _fail(
                                    "kinematics",
                                    f"kinematic-track cache {cache_target} "
                                    f"holds {len(stored_times)} resumable "
                                    "frame(s) that do not exactly match the "
                                    "native PTS prefix; pass --force to "
                                    "restart or choose a distinct --cache "
                                    "path.",
                                )
                            cached = list(snapshot.frames)
            elif cache_target.is_file() and force_flag:
                try:
                    world_module.load_world_cache(cache_target)
                except (
                    world_module.CacheCorruptError,
                    world_module.CacheStaleError,
                ):
                    _quarantine_quietly(cache_target)
                except world_module.CacheError:
                    try:
                        cache_target.unlink()
                    except OSError as exc:
                        raise _fail(
                            "kinematics",
                            f"could not replace kinematic-track cache "
                            f"{cache_target}: {exc}.",
                        ) from exc
                else:
                    try:
                        cache_target.unlink()
                    except OSError as exc:
                        raise _fail(
                            "kinematics",
                            f"could not replace kinematic-track cache "
                            f"{cache_target}: {exc}.",
                        ) from exc
                cached = []

            if cache_hit:
                observations = list(cached)
                inferred = 0
            else:
                if cached:
                    if len(cached) > total:
                        raise _fail(
                            "kinematics",
                            f"kinematic-track cache {cache_target} holds more "
                            f"frames ({len(cached)}) than the {total} native "
                            f"frame(s) in [{req_start!r}, {req_end!r}); "
                            "refusing to reuse.",
                        )
                    remaining = tuple(expected[len(cached):])
                else:
                    remaining = expected
                if not remaining:
                    try:
                        world_module.write_complete_world_cache(
                            cache_target, identity, cached
                        )
                    except world_module.CacheError as exc:
                        raise _fail(
                            "kinematics",
                            f"could not write kinematic-track cache to "
                            f"{cache_target}: {exc}.",
                        ) from exc
                    observations = list(cached)
                    inferred = 0
                else:
                    if native_frame_factory is not None:
                        try:
                            frame_iter = native_frame_factory(
                                remaining, metadata
                            )
                        except (AnalyzeServeError, AnalyzeServeCancelled):
                            raise
                        except Exception as exc:
                            raise _fail(
                                "kinematics",
                                f"could not stream native frames: {exc}.",
                            ) from exc
                    else:
                        try:
                            frame_iter = frames_module.iter_native_frames(
                                video_path,
                                float(remaining[0]),
                                req_end,
                                source=metadata,
                                ffmpeg=ffmpeg_exe,
                                ffprobe=ffprobe_exe,
                                is_cancelled=is_cancelled,
                                filter_expression=color_filter,
                            )
                        except frames_module.FrameCancelled as exc:
                            raise AnalyzeServeCancelled(
                                "kinematics",
                                f"kinematic extraction was cancelled: {exc}.",
                            ) from exc
                        except frames_module.FrameError as exc:
                            raise _fail(
                                "kinematics",
                                f"could not decode native frames: {exc}.",
                            ) from exc
                    try:
                        iterator = iter(frame_iter)
                    except TypeError as exc:
                        raise _fail(
                            "kinematics",
                            f"native frame source is not iterable: {exc}.",
                        ) from exc
                    all_obs: list[WorldFrameObservation] = list(cached)
                    inferred = 0
                    last_obs_time: float | None = (
                        float(cached[-1].time_seconds) if cached else None
                    )

                    def _flush_partial() -> None:
                        try:
                            world_module.write_partial_world_cache(
                                cache_target, identity, all_obs
                            )
                        except world_module.CacheError as exc:
                            raise _fail(
                                "kinematics",
                                "could not write kinematic-track cache to "
                                f"{cache_target}: {exc}.",
                            ) from exc

                    try:
                        if _cancelled():
                            raise AnalyzeServeCancelled(
                                "kinematics",
                                "kinematic extraction was cancelled before "
                                "starting.",
                            )
                        position = 0
                        while True:
                            try:
                                frame = next(iterator)
                            except StopIteration:
                                break
                            except AnalyzeServeCancelled:
                                raise
                            except AnalyzeServeError:
                                raise
                            except frames_module.FrameCancelled as exc:
                                if inferred:
                                    _flush_partial()
                                raise AnalyzeServeCancelled(
                                    "kinematics",
                                    "kinematic extraction was cancelled: "
                                    f"{exc}.",
                                ) from exc
                            except Exception as exc:
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "could not decode native frames from "
                                    f"{video_path}: {exc}.",
                                ) from exc
                            if _cancelled():
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise AnalyzeServeCancelled(
                                    "kinematics",
                                    "kinematic extraction was cancelled after "
                                    f"{inferred} of {total} frames.",
                                )
                            try:
                                frame_time = float(frame.time_seconds)  # type: ignore[union-attr]
                                frame_image = frame.image  # type: ignore[union-attr]
                            except (AttributeError, TypeError, ValueError) as exc:
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "native frame source yielded a malformed "
                                    f"frame: {exc}.",
                                ) from exc
                            if (
                                position >= len(remaining)
                                or frame_time != remaining[position]
                            ):
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                want = (
                                    remaining[position]
                                    if position < len(remaining)
                                    else None
                                )
                                raise _fail(
                                    "kinematics",
                                    "native frame stream diverged from the "
                                    "expected PTS grid: got "
                                    f"t={frame_time!r}, expected t={want!r}; "
                                    "refusing to emit misaligned rows.",
                                )
                            try:
                                observation = infer_world(frame_image, frame_time)
                            except AnalyzeServeCancelled:
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise
                            except Exception as exc:
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "pose inference failed at "
                                    f"t={frame_time}s: {exc}.",
                                ) from exc
                            if not isinstance(
                                observation, WorldFrameObservation
                            ):
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "pose backend returned "
                                    f"{type(observation).__name__}; expected "
                                    "WorldFrameObservation (sparse 2D rows "
                                    "are never valid kinematic-track rows).",
                                )
                            try:
                                obs_time = float(observation.time_seconds)
                            except (AttributeError, TypeError, ValueError) as exc:
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "pose backend returned a malformed "
                                    f"observation: {exc}.",
                                ) from exc
                            if obs_time != frame_time:
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "pose backend altered canonical time: "
                                    f"frame t={frame_time!r} produced "
                                    f"observation t={obs_time!r}; canonical "
                                    "source time must be preserved exactly.",
                                )
                            if (
                                last_obs_time is not None
                                and not obs_time > last_obs_time
                            ):
                                if inferred:
                                    try:
                                        _flush_partial()
                                    except AnalyzeServeError:
                                        pass
                                raise _fail(
                                    "kinematics",
                                    "kinematic-track observations must be "
                                    "strictly increasing: last "
                                    f"{last_obs_time!r}, new {obs_time!r}.",
                                )
                            last_obs_time = obs_time
                            all_obs.append(observation)
                            inferred += 1
                            position += 1
                            _progress(
                                f"analyze-serve: kinematics "
                                f"{len(cached) + inferred}/{total} frames"
                            )
                            if inferred % CACHE_FLUSH_INTERVAL_FRAMES == 0:
                                _flush_partial()
                        if position != len(remaining):
                            if inferred:
                                try:
                                    _flush_partial()
                                except AnalyzeServeError:
                                    pass
                            raise _fail(
                                "kinematics",
                                f"native frame stream ended after {position} "
                                f"of {len(remaining)} expected frame(s); "
                                "refusing to emit a truncated cache.",
                            )
                    except AnalyzeServeCancelled:
                        if inferred:
                            try:
                                _flush_partial()
                            except AnalyzeServeError:
                                pass
                        raise
                    except AnalyzeServeError:
                        if inferred and not cache_target.is_file():
                            try:
                                _flush_partial()
                            except AnalyzeServeError:
                                pass
                        raise
                    if tuple(frame.time_seconds for frame in all_obs) != tuple(
                        expected
                    ):
                        raise _fail(
                            "kinematics",
                            "kinematic-track frame times do not exactly "
                            "match the native PTS grid; refusing to publish "
                            "a misaligned cache.",
                        )
                    try:
                        world_module.write_complete_world_cache(
                            cache_target, identity, all_obs
                        )
                    except world_module.CacheError as exc:
                        raise _fail(
                            "kinematics",
                            "could not write kinematic-track cache to "
                            f"{cache_target}: {exc}.",
                        ) from exc
                    observations = list(all_obs)
        finally:
            closer = getattr(backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

        # --- raw ball/racket observations on the same exact PTS grid ---
        rv_config = racketvision_config or RacketVisionConfig()
        if not isinstance(rv_config, RacketVisionConfig):
            raise _fail("tracking", "invalid racketvision_config.")
        try:
            rv_config = rv_config.resolved_device()
        except Exception as exc:
            raise _fail("tracking", str(exc)) from exc
        try:
            rv_identity = RacketVisionCacheIdentity(
                source_fingerprint=fingerprint,
                attempt_start_seconds=req_start,
                attempt_end_seconds=req_end,
                timeline_fingerprint=fingerprint_frame_times(expected),
                tracker_fingerprint=(
                    racketvision_tracker_fingerprint.strip()
                    if racketvision_tracker_fingerprint is not None
                    else fingerprint_racketvision_config(rv_config)
                ),
            )
        except Exception as exc:
            raise _fail(
                "tracking", f"could not identify RacketVision inputs: {exc}."
            ) from exc

        racketvision_hit = False
        racketvision_observations = None
        if racketvision_cache_target.is_file() and not force_flag:
            try:
                rv_snapshot = load_racketvision_cache(
                    racketvision_cache_target,
                    rv_identity,
                    expected_times=expected,
                )
            except (RacketVisionCacheCorruptError, RacketVisionCacheStaleError):
                _quarantine_quietly(racketvision_cache_target)
            except RacketVisionCacheError as exc:
                raise _fail("tracking", f"could not read object track: {exc}.") from exc
            else:
                racketvision_observations = rv_snapshot.observations
                racketvision_hit = True

        if racketvision_observations is None:
            try:
                frames = (
                    racketvision_frame_factory(expected, metadata)
                    if racketvision_frame_factory is not None
                    else iter_racketvision_model_frames(
                        video_path,
                        expected,
                        source=metadata,
                        ffmpeg=ffmpeg_exe,
                        ffprobe=ffprobe_exe,
                        is_cancelled=is_cancelled,
                        filter_expression=color_filter,
                    )
                )
                tracker = (
                    racketvision_tracker_factory(rv_config)
                    if racketvision_tracker_factory is not None
                    else RacketVisionTracker(rv_config)
                )
                racketvision_observations = tuple(tracker.track_frames(frames))
                if tuple(
                    observation.time_seconds
                    for observation in racketvision_observations
                ) != tuple(expected):
                    raise RacketVisionCacheError(
                        "tracker output does not match the native PTS grid."
                    )
                write_racketvision_cache(
                    racketvision_cache_target,
                    rv_identity,
                    racketvision_observations,
                    expected_times=expected,
                )
            except Exception as exc:
                raise _fail("tracking", f"RacketVision tracking failed: {exc}.") from exc

        # --- 3D waveform chain with optional audio cue (never 2D, never sparse) ---
        _progress("analyze-serve: building 3D waveforms")
        if _cancelled():
            raise AnalyzeServeCancelled(
                "solve", "analyze-serve was cancelled before waveform analysis."
            )
        try:
            filtered = filter_module.build_filtered_world_track(
                tuple(observations), cfg_filter
            )
        except Exception as exc:
            raise _fail(
                "solve", f"3D filtering failed: {exc}."
            ) from exc
        # --- raw-source audio demux once, aligned to exact kinematic PTS ---
        # Nonfatal: demux/alignment failure or an absent stream yields
        # honestly unavailable audio channels.
        _progress("analyze-serve: sampling audio cue")
        audio_status = "unavailable"
        aligned_audio: Any = None
        try:
            if _cancelled():
                raise AnalyzeServeCancelled(
                    "solve",
                    "analyze-serve was cancelled before audio sampling.",
                )
            if audio_energies_fn is not None:
                custom = audio_energies_fn(video_path, tuple(expected))
                if custom is None:
                    aligned_audio = None
                else:
                    aligned_audio = list(custom)  # type: ignore[arg-type]
                    if len(aligned_audio) != len(expected):
                        aligned_audio = None
                    else:
                        audio_status = "present"
                        if not aligned_audio:
                            aligned_audio = None
                            audio_status = "unavailable"
            else:
                dense_times = audio_module.dense_audio_schedule(duration)
                dense_iter = audio_module.iter_audio_energy(
                    video_path,
                    dense_times,
                    ffmpeg=ffmpeg_exe,
                    ffprobe=ffprobe_exe,
                    is_cancelled=is_cancelled,
                )
                dense_energies = list(dense_iter)
                pooled = audio_module.align_audio_maxpool(
                    tuple(dense_energies), tuple(expected)
                )
                aligned_audio = list(pooled)
                audio_status = "present"
        except AnalyzeServeCancelled:
            raise
        except audio_module.AudioCancelled as exc:
            raise AnalyzeServeCancelled(
                "solve", f"audio sampling was cancelled: {exc}."
            ) from exc
        except Exception:
            aligned_audio = None
            audio_status = "unavailable"
        # Shared-policy explicit flags (never local maxima): qualify the
        # aligned energies with the M3 session-relative policy at
        # DecoderConfig defaults so the waveform flag channel carries
        # the shared contract instead of derived local peaks.
        if aligned_audio is not None:
            try:
                aligned_audio = _attach_shared_transient_flags(
                    aligned_audio, tuple(expected)
                )
            except AnalyzeServeCancelled:
                raise
            except Exception:
                aligned_audio = None
                audio_status = "unavailable"
        try:
            scene_audio, scene_transients = _scene_audio_rows(
                aligned_audio, tuple(expected)
            )
            scene_track = build_scene_track_with_racketvision_observations(
                world_module.WorldCacheSnapshot(
                    identity=identity,
                    frames=tuple(observations),
                    complete=True,
                ),
                racketvision_observations,
                audio=scene_audio,
                audio_transients=scene_transients,
            )
        except Exception as exc:
            raise _fail("tracking", f"could not build SceneTrack: {exc}.") from exc
        waveform_audio = (
            [
                None
                if frame.audio_energy is None
                else (
                    frame.audio_energy,
                    None
                    if frame.audio_transient is None
                    else float(frame.audio_transient),
                )
                for frame in scene_track.frames
            ]
            if scene_audio is not None
            else None
        )
        try:
            track = waveforms_module.build_kinematic_waveform_track(
                filtered, cfg_waveforms, audio_energies=waveform_audio
            )
        except Exception as exc:
            if aligned_audio is not None:
                try:
                    track = waveforms_module.build_kinematic_waveform_track(
                        filtered, cfg_waveforms, audio_energies=None
                    )
                except Exception as exc2:
                    raise _fail(
                        "solve", f"3D waveform construction failed: {exc2}."
                    ) from exc2
                aligned_audio = None
                audio_status = "unavailable"
            else:
                raise _fail(
                    "solve", f"3D waveform construction failed: {exc}."
                ) from exc
        try:
            scene_features = scene_features_module.build_scene_feature_series(
                scene_track
            )
            anchor_set = composite_module.build_composite_anchor_set(
                track, cfg_composite, scene_features=scene_features
            )
        except Exception as exc:
            raise _fail(
                "solve", f"composite candidate scoring failed: {exc}."
            ) from exc
        if not isinstance(anchor_set, CompositeAnchorSet):
            raise _fail(
                "solve",
                "composite candidate builder returned "
                f"{type(anchor_set).__name__}; expected CompositeAnchorSet.",
            )

        # --- six-anchor DP selection (eligible = coverage > 0) ---
        _progress("analyze-serve: solving six-anchor chronology")
        if _cancelled():
            raise AnalyzeServeCancelled(
                "solve", "analyze-serve was cancelled before DP selection."
            )
        eligible: dict[str, list[CompositeAnchorCandidate]] = {}
        for stage in SIX_ANCHOR_STAGES:
            dense = list(anchor_set.for_stage(stage))
            eligible[stage] = [c for c in dense if float(c.coverage) > 0.0]
        try:
            solution = six_anchor_module.solve_six_anchors(eligible, cfg_solver)
        except Exception as exc:
            raise _fail("solve", f"six-anchor DP selection failed: {exc}.") from exc
        selected: dict[str, CompositeAnchorCandidate | None] = {}
        for stage in SIX_ANCHOR_STAGES:
            index = solution.selected_index(stage)
            selected[stage] = (
                eligible[stage][index] if index is not None else None
            )

        # --- derived midpoints (exact, after anchor selection) ---
        def _midpoint(
            first: CompositeAnchorCandidate | None,
            second: CompositeAnchorCandidate | None,
        ) -> float | None:
            if first is None or second is None:
                return None
            return (float(first.time_seconds) + float(second.time_seconds)) / 2.0

        mid_acceleration = _midpoint(selected["cocking"], selected["contact"])
        mid_deceleration = _midpoint(selected["contact"], selected["finish"])

        # Waveform support at derived midpoints (honest wrist evidence).
        sample_times = [float(sample.time_seconds) for sample in track.samples]
        wrist_speed_idx = waveforms_module.CHANNEL_INDEX["right_wrist_speed"]

        def _midpoint_support(mid: float | None) -> tuple[bool, float]:
            if mid is None or not sample_times:
                return (False, 0.0)
            nearest = min(
                range(len(sample_times)),
                key=lambda i: abs(sample_times[i] - mid),
            )
            sample = track.samples[nearest]
            value = sample.channel_values[wrist_speed_idx]
            available = bool(sample.channel_available[wrist_speed_idx])
            if available and value is not None and math.isfinite(float(value)):
                uncertainties = [
                    float(u) for u in sample.channel_uncertainty_seconds
                ]
                return (True, max(uncertainties) if uncertainties else 0.0)
            return (False, 0.0)

        acc_supported, acc_sample_unc = _midpoint_support(mid_acceleration)
        dec_supported, dec_sample_unc = _midpoint_support(mid_deceleration)

        # --- attempt range (effective): requested range widened only to
        # include the exact native grid support (ffprobe lower-bound
        # tolerance can place the first grid PTS marginally below start).
        grid_start = min(float(observations[0].time_seconds), req_start)
        try:
            attempt_range = MediaRange(grid_start, req_end)
        except Exception as exc:
            raise _fail(
                "solve", f"could not build serve-001 range: {exc}."
            ) from exc

        # --- stage emission in canonical order ---
        eps = _POINT_INTERVAL_EPSILON_SECONDS
        ordered_times: list[float] = []
        time_by_stage: dict[str, float | None] = {}
        for stage in SIX_ANCHOR_STAGES:
            candidate = selected[stage]
            time_by_stage[stage] = (
                float(candidate.time_seconds) if candidate is not None else None
            )
        time_by_stage["acceleration"] = mid_acceleration
        time_by_stage["deceleration"] = mid_deceleration
        for stage in STAGE_ORDER:
            moment = time_by_stage.get(stage)
            if moment is not None:
                ordered_times.append(float(moment))
        # Chronology guard: selected anchors strictly increase by DP
        # feasibility and midpoints lie strictly between their bounds.
        for earlier, later in zip(ordered_times, ordered_times[1:]):
            if not later > earlier:
                raise _fail(
                    "solve",
                    "selected stage times are not strictly chronological "
                    f"({earlier!r} followed by {later!r}); refusing to emit "
                    "an inconsistent phase.",
                )

        def _interval_for(moment: float, next_moment: float | None) -> MediaRange:
            upper = attempt_range.end_seconds
            if next_moment is not None:
                upper = min(upper, float(next_moment))
            end = min(float(moment) + eps, upper)
            if not end > float(moment):
                # Distinct selected times guarantee end > moment unless
                # the range end coincides with the keyframe; clamp to
                # the range end which still exceeds moment here only
                # when moment < range end (always true: grid < req_end).
                end = attempt_range.end_seconds
            try:
                return MediaRange(float(moment), end)
            except Exception as exc:
                raise _fail(
                    "solve", f"could not build stage interval: {exc}."
                ) from exc

        # Map each available stage to the next available stage time for
        # ordering-safe linkage intervals.
        available_in_order = [
            (stage, float(time_by_stage[stage]))
            for stage in STAGE_ORDER
            if time_by_stage.get(stage) is not None
        ]
        next_by_stage: dict[str, float | None] = {}
        for position, (stage, _) in enumerate(available_in_order):
            if position + 1 < len(available_in_order):
                next_by_stage[stage] = available_in_order[position + 1][1]
            else:
                next_by_stage[stage] = None

        stages: dict[str, StagePhase] = {}
        anomalies: list[str] = []
        for stage in STAGE_ORDER:
            if stage in SIX_ANCHOR_STAGES:
                candidate = selected[stage]
                if candidate is None:
                    if not eligible[stage]:
                        limitations = ("no_candidate_available",)
                        anomaly = f"no_candidate_{stage}"
                    else:
                        limitations = ("skipped_for_global_consistency",)
                        anomaly = f"skipped_{stage}_for_consistency"
                    stages[stage] = StagePhase(
                        availability="unavailable",
                        provenance="body_pose",
                        confidence=0.0,
                        interval=None,
                        keyframe_seconds=None,
                        temporal_uncertainty_seconds=None,
                        evidence=(),
                        limitations=limitations,
                    )
                    anomalies.append(anomaly)
                    continue
                assert isinstance(candidate.cue_values, Mapping)
                cue_order = composite_module.COMPOSITE_CUE_NAMES[stage]
                evidence_tokens = tuple(
                    cue
                    for cue in cue_order
                    if candidate.cue_values.get(cue) is not None
                )
                limitations: tuple[str, ...] = ()
                provenance = "body_pose"
                if stage == "contact":
                    audio_cue = candidate.cue_values.get("audio_transient")
                    if (
                        audio_cue is not None
                        and not isinstance(audio_cue, bool)
                        and math.isfinite(float(audio_cue))
                        and float(audio_cue) > 0.0
                    ):
                        provenance = "body_pose_audio"
                    else:
                        provenance = "body_pose"
                moment = float(candidate.time_seconds)
                stages[stage] = StagePhase(
                    availability="available",
                    provenance=provenance,
                    confidence=max(0.0, min(1.0, float(candidate.score))),
                    interval=_interval_for(moment, next_by_stage.get(stage)),
                    keyframe_seconds=moment,
                    temporal_uncertainty_seconds=float(
                        candidate.temporal_uncertainty_seconds
                    ),
                    evidence=evidence_tokens,
                    limitations=limitations,
                )
            else:
                # Derived midpoint stages.
                if stage == "acceleration":
                    mid = mid_acceleration
                    first_c = selected["cocking"]
                    second_c = selected["contact"]
                    supported = acc_supported
                    sample_unc = acc_sample_unc
                    token = "midpoint_of_cocking_and_contact"
                else:
                    mid = mid_deceleration
                    first_c = selected["contact"]
                    second_c = selected["finish"]
                    supported = dec_supported
                    sample_unc = dec_sample_unc
                    token = "midpoint_of_contact_and_finish"
                if mid is None or first_c is None or second_c is None:
                    stages[stage] = StagePhase(
                        availability="unavailable",
                        provenance="body_pose",
                        confidence=0.0,
                        interval=None,
                        keyframe_seconds=None,
                        temporal_uncertainty_seconds=None,
                        evidence=(),
                        limitations=("bounding_anchor_unavailable",),
                    )
                    anomalies.append(f"no_midpoint_{stage}")
                    continue
                confidence = (
                    max(0.0, min(1.0, float(first_c.score)))
                    + max(0.0, min(1.0, float(second_c.score)))
                ) / 2.0
                uncertainty = max(
                    float(first_c.temporal_uncertainty_seconds),
                    float(second_c.temporal_uncertainty_seconds),
                    float(sample_unc),
                )
                evidence_tokens = (token,)
                if supported:
                    evidence_tokens = evidence_tokens + (
                        "midpoint_wrist_waveform_support",
                    )
                stages[stage] = StagePhase(
                    availability="available",
                    provenance="body_pose",
                    confidence=confidence,
                    interval=_interval_for(float(mid), next_by_stage.get(stage)),
                    keyframe_seconds=float(mid),
                    temporal_uncertainty_seconds=uncertainty,
                    evidence=evidence_tokens,
                    limitations=(),
                )

        status = _structural_status(
            [stages[key].availability for key in STAGE_ORDER]
        )
        try:
            attempt_phase = AttemptPhase(
                attempt_id=ANALYZE_SERVE_ATTEMPT_ID,
                attempt_range=MediaRange(
                    start_seconds=attempt_range.start_seconds,
                    end_seconds=attempt_range.end_seconds,
                ),
                method_version=ANALYZE_SERVE_METHOD_VERSION,
                config_id=cfg_solver.config_id,
                stages=stages,
                structural_status=status,
                anomalies=tuple(anomalies),
            )
        except Exception as exc:
            raise _fail(
                "solve", f"could not build serve-001 phase: {exc}."
            ) from exc
        try:
            serve_fingerprint = build_serve_fingerprint_v1(
                track,
                attempt_phase,
                source_fingerprint=fingerprint,
                body_model_name=model_name,
                body_model_version=model_version,
            )
            fingerprint_text = serve_fingerprint.to_json()
        except Exception as exc:
            raise _fail("solve", f"could not build ServeFingerprintV1: {exc}.") from exc
        try:
            phase_document = PhaseDocument(
                source_fingerprint=fingerprint,
                source_duration_seconds=duration,
                attempts=(attempt_phase,),
            )
        except Exception as exc:
            raise _fail(
                "solve", f"could not build checkpoints document: {exc}."
            ) from exc

        # --- deterministic 3D diagnostic JSON ---
        def _selected_record(
            candidate: CompositeAnchorCandidate | None,
        ) -> dict[str, Any]:
            if candidate is None:
                return {
                    "coverage": None,
                    "cue_values": None,
                    "score": None,
                    "status": "skipped",
                    "temporal_uncertainty_seconds": None,
                    "time_seconds": None,
                }
            assert isinstance(candidate.cue_values, Mapping)
            return {
                "coverage": float(candidate.coverage),
                "cue_values": {
                    cue: candidate.cue_values[cue]
                    for cue in composite_module.COMPOSITE_CUE_NAMES[
                        candidate.stage
                    ]
                },
                "score": float(candidate.score),
                "status": "selected",
                "temporal_uncertainty_seconds": float(
                    candidate.temporal_uncertainty_seconds
                ),
                "time_seconds": float(candidate.time_seconds),
            }

        visual_diagnostics = None
        if scene_track is not None:
            visual_diagnostics = build_scene_visual_diagnostics(
                scene_track,
                selected_release_time_seconds=time_by_stage.get("release"),
                selected_contact_time_seconds=time_by_stage.get("contact"),
            )

        diagnostics = {
            "attempt_id": ANALYZE_SERVE_ATTEMPT_ID,
            "attempt_range": attempt_range.to_dict(),
            "audio": audio_status,
            "racketvision": {
                "cache_hit": racketvision_hit,
                "cache_path": str(racketvision_cache_target),
            },
            "candidates": {
                stage: {
                    "eligible": len(eligible[stage]),
                    "total": len(list(anchor_set.for_stage(stage))),
                }
                for stage in SIX_ANCHOR_STAGES
            },
            "configs": {
                "composite": {
                    "config_id": cfg_composite.config_id,
                    "method_version": composite_module.COMPOSITE_ANCHORS_METHOD_VERSION,
                },
                "filter": {
                    "config_id": cfg_filter.config_id,
                    "method_version": filter_module.WORLD_FILTER_METHOD_VERSION,
                },
                "six_anchor_dp": {
                    "config_id": cfg_solver.config_id,
                    "method_version": six_anchor_module.SIX_ANCHOR_METHOD_VERSION,
                },
                "solver_transition": {
                    "config_id": cfg_solver.config_id,
                    "method_version": "phase-solver-v1",
                },
                "waveforms": {
                    "config_id": cfg_waveforms.config_id,
                    "method_version": waveforms_module.KINEMATIC_WAVEFORMS_METHOD_VERSION,
                },
            },
            "contact_note": (
                "contact uses body_pose_audio provenance when the selected "
                "contact carries a strictly positive audio_transient cue "
                "(> 0), otherwise body_pose; a merely available 0.0 flag "
                "never qualifies; audio channels are unavailable without "
                "present raw-source audio."
            ),
            "coordinate_2d": "not_used",
            "derived": {
                "acceleration": {
                    "bounding_anchors": ["cocking", "contact"],
                    "midpoint_of": [
                        time_by_stage.get("cocking"),
                        time_by_stage.get("contact"),
                    ],
                    "status": (
                        "selected"
                        if mid_acceleration is not None
                        and selected["cocking"] is not None
                        and selected["contact"] is not None
                        else "skipped"
                    ),
                    "time_seconds": mid_acceleration,
                    "wrist_waveform_support": bool(acc_supported),
                },
                "deceleration": {
                    "bounding_anchors": ["contact", "finish"],
                    "midpoint_of": [
                        time_by_stage.get("contact"),
                        time_by_stage.get("finish"),
                    ],
                    "status": (
                        "selected"
                        if mid_deceleration is not None
                        and selected["contact"] is not None
                        and selected["finish"] is not None
                        else "skipped"
                    ),
                    "time_seconds": mid_deceleration,
                    "wrist_waveform_support": bool(dec_supported),
                },
            },
            "frame_count": len(observations),
            "method_version": ANALYZE_SERVE_METHOD_VERSION,
            "model": {"name": model_name, "version": model_version},
            "orchestration_version": ANALYZE_SERVE_VERSION,
            "pts_seconds": list(expected),
            "requested_range": requested_range.to_dict(),
            "schema_version": ANALYZE_SERVE_SCHEMA_VERSION,
            "selected": {
                stage: _selected_record(selected[stage])
                for stage in SIX_ANCHOR_STAGES
            },
            "source_duration_seconds": duration,
            "source_fingerprint": fingerprint,
            "total_score": float(solution.total_score),
        }
        if visual_diagnostics is not None:
            diagnostics["visual_evidence"] = visual_diagnostics
        diagnostics_text = json.dumps(diagnostics, sort_keys=True, indent=2) + "\n"

        # --- collision check before staging review outputs ---
        if _cancelled():
            raise AnalyzeServeCancelled(
                "review", "analyze-serve was cancelled before review rendering."
            )
        if checkpoints_out.exists() and not force_flag:
            raise _fail(
                "write",
                f"output collision: {checkpoints_out} already exists. Pass "
                f"--force to replace outputs in {session_dir}. ",
            )
        if diagnostics_out.exists() and not force_flag:
            raise _fail(
                "write",
                f"output collision: {diagnostics_out} already exists. Pass "
                f"--force to replace outputs in {session_dir}. ",
            )
        if fingerprint_out.exists() and not force_flag:
            raise _fail(
                "write",
                f"output collision: {fingerprint_out} already exists. Pass "
                f"--force to replace outputs in {session_dir}. ",
            )
        if review_dir.exists() and not force_flag:
            raise _fail(
                "review",
                f"output collision: {review_dir} already exists. Pass "
                "--force to replace the review directory. ",
            )
        if force_flag:
            for path in (checkpoints_out, diagnostics_out, fingerprint_out):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                if review_dir.exists():
                    shutil.rmtree(review_dir)
            except OSError as exc:
                raise _fail(
                    "review",
                    f"could not replace review directory {review_dir}: {exc}. ",
                ) from exc

        # --- clean source-frame review (no caption or overlay) ---
        _progress("analyze-serve: rendering review frames")
        render_times = sorted(
            {
                float(time_by_stage[stage])
                for stage in STAGE_ORDER
                if time_by_stage.get(stage) is not None
            }
        )
        actual_by_requested: dict[float, float] = {}
        frames_by_time: dict[float, Any] = {}
        if render_times:
            for moment in render_times:
                if not (
                    math.isfinite(moment)
                    and 0 <= moment < duration + 1e-9
                ):
                    raise _fail(
                        "review",
                        f"selected timestamp {moment!r} lies outside the "
                        "source timeline; refusing to render.",
                    )
            try:
                if sample_frame_fn is not None:
                    sampled_list = [
                        sample_frame_fn(video_path, moment, metadata)
                        for moment in render_times
                    ]
                else:
                    iterator = frames_module.iter_sampled_frames(
                        video_path,
                        tuple(render_times),
                        source=metadata,
                        ffmpeg=ffmpeg_exe,
                        ffprobe=ffprobe_exe,
                        is_cancelled=is_cancelled,
                        filter_expression=color_filter,
                    )
                    sampled_list = list(iterator)
            except AnalyzeServeCancelled:
                raise
            except AnalyzeServeError:
                raise
            except frames_module.FrameCancelled as exc:
                raise AnalyzeServeCancelled(
                    "review", f"review sampling was cancelled: {exc}."
                ) from exc
            except frames_module.FrameError as exc:
                raise _fail(
                    "review", f"could not sample source frames: {exc}."
                ) from exc
            except Exception as exc:
                raise _fail(
                    "review", f"could not sample source frames: {exc}."
                ) from exc
            if len(sampled_list) != len(render_times):
                raise _fail(
                    "review",
                    f"source sampling returned {len(sampled_list)} frame(s) "
                    f"for {len(render_times)} timestamp(s); refusing to "
                    "publish mislinked review images.",
                )
            for requested, sampled in zip(render_times, sampled_list):
                if _cancelled():
                    raise AnalyzeServeCancelled(
                        "review",
                        "analyze-serve was cancelled during review rendering.",
                    )
                try:
                    actual = float(sampled.time_seconds)  # type: ignore[union-attr]
                    image = np.ascontiguousarray(
                        sampled.image, dtype=np.uint8  # type: ignore[union-attr]
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise _fail(
                        "review",
                        f"sampled frame at t={requested!r} is malformed: "
                        f"{exc}.",
                    ) from exc
                actual_by_requested[requested] = actual
                frames_by_time[requested] = image

        manual_actual_by_requested: dict[float, float] = {}
        manual_frames_by_time: dict[float, Any] = {}
        manual_render_times: list[float] = []
        if anchor2_flag:
            assert manual_by_stage is not None
            manual_render_times = sorted(
                {
                    float(value)
                    for value in manual_by_stage.values()
                    if value is not None
                }
            )
            for moment in manual_render_times:
                if not (
                    math.isfinite(moment)
                    and 0 <= moment < duration + 1e-9
                ):
                    raise _fail(
                        "review",
                        f"manual anchor2 timestamp {moment!r} lies outside "
                        "the source timeline; refusing to render.",
                    )
            if manual_render_times:
                try:
                    if sample_frame_fn is not None:
                        manual_sampled = [
                            sample_frame_fn(video_path, moment, metadata)
                            for moment in manual_render_times
                        ]
                    else:
                        manual_iter = frames_module.iter_sampled_frames(
                            video_path,
                            tuple(manual_render_times),
                            source=metadata,
                            ffmpeg=ffmpeg_exe,
                            ffprobe=ffprobe_exe,
                            is_cancelled=is_cancelled,
                            filter_expression=color_filter,
                        )
                        manual_sampled = list(manual_iter)
                except AnalyzeServeCancelled:
                    raise
                except AnalyzeServeError:
                    raise
                except frames_module.FrameCancelled as exc:
                    raise AnalyzeServeCancelled(
                        "review", f"review sampling was cancelled: {exc}."
                    ) from exc
                except frames_module.FrameError as exc:
                    raise _fail(
                        "review", f"could not sample source frames: {exc}."
                    ) from exc
                except Exception as exc:
                    raise _fail(
                        "review", f"could not sample source frames: {exc}."
                    ) from exc
                if len(manual_sampled) != len(manual_render_times):
                    raise _fail(
                        "review",
                        f"source sampling returned {len(manual_sampled)} "
                        f"frame(s) for {len(manual_render_times)} manual "
                        "timestamp(s); refusing to publish mislinked "
                        "review images.",
                    )
                for requested, sampled in zip(
                    manual_render_times, manual_sampled
                ):
                    if _cancelled():
                        raise AnalyzeServeCancelled(
                            "review",
                            "analyze-serve was cancelled during review "
                            "rendering.",
                        )
                    try:
                        actual = float(sampled.time_seconds)  # type: ignore[union-attr]
                        image = np.ascontiguousarray(
                            sampled.image, dtype=np.uint8  # type: ignore[union-attr]
                        )
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise _fail(
                            "review",
                            f"sampled manual frame at t={requested!r} is "
                            f"malformed: {exc}.",
                        ) from exc
                    manual_actual_by_requested[requested] = actual
                    manual_frames_by_time[requested] = image

        try:
            parent = review_dir.parent
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _fail(
                "review",
                f"could not create output directory {parent}: {exc}.",
            ) from exc
        try:
            staging = Path(
                tempfile.mkdtemp(
                    prefix=review_dir.name + ".tmp-", dir=str(parent)
                )
            )
        except OSError as exc:
            raise _fail(
                "review", f"could not stage review output: {exc}."
            ) from exc

        def _remove_staging() -> None:
            try:
                shutil.rmtree(staging, ignore_errors=True)
            except OSError:
                pass

        entries: list[dict[str, Any]] = []
        image_count = 0
        try:
            for stage in STAGE_ORDER:
                if _cancelled():
                    raise AnalyzeServeCancelled(
                        "review",
                        "analyze-serve was cancelled during review rendering.",
                    )
                phase = stages[stage]
                requested = time_by_stage.get(stage)
                base: dict[str, Any] = {
                    "actual_source_time": None,
                    "attempt_id": ANALYZE_SERVE_ATTEMPT_ID,
                    "availability": phase.availability,
                    "confidence": float(phase.confidence),
                    "image": None,
                    "provenance": phase.provenance,
                    "reason": None,
                    "requested_source_time": requested,
                    "stage": stage,
                    "status": "unavailable",
                }
                if phase.availability == "unavailable" or requested is None:
                    base["reason"] = "stage_unavailable"
                    if anchor2_flag:
                        assert manual_by_stage is not None
                        manual_requested = manual_by_stage.get(stage)
                        base["manual_requested_source_time"] = manual_requested
                        base["manual_actual_source_time"] = None
                        base["manual_image"] = None
                        base["delta_selected_minus_manual_ms"] = None
                        if manual_requested is not None:
                            manual_f = float(manual_requested)
                            if manual_f not in manual_frames_by_time:
                                raise _fail(
                                    "review",
                                    f"no sampled manual frame for {stage} at "
                                    f"t={manual_f!r}; refusing to publish "
                                    "mislinked review images.",
                                )
                            manual_actual = manual_actual_by_requested[manual_f]
                            manual_image = manual_frames_by_time[manual_f]
                            manual_rel = f"manual-{_safe_component(stage)}.jpg"
                            manual_dest = staging / manual_rel
                            try:
                                if encode_jpeg_fn is not None:
                                    encode_jpeg_fn(manual_image, manual_dest)
                                else:
                                    _encode_jpeg_ffmpeg(
                                        manual_image, manual_dest, ffmpeg_exe
                                    )
                            except AnalyzeServeCancelled:
                                raise
                            except AnalyzeServeError:
                                raise
                            except Exception as exc:
                                raise _fail(
                                    "review",
                                    f"could not write review image "
                                    f"{manual_dest}: {exc}.",
                                ) from exc
                            base["manual_actual_source_time"] = manual_actual
                            base["manual_image"] = manual_rel
                            image_count += 1
                    entries.append(base)
                    continue
                requested_f = float(requested)
                if requested_f not in frames_by_time:
                    raise _fail(
                        "review",
                        f"no sampled frame for selected {stage} at "
                        f"t={requested_f!r}; refusing to publish mislinked "
                        "review images.",
                    )
                actual = actual_by_requested[requested_f]
                image = frames_by_time[requested_f]
                rel = f"{_safe_component(stage)}.jpg"
                dest = staging / rel
                try:
                    if encode_jpeg_fn is not None:
                        encode_jpeg_fn(image, dest)
                    else:
                        _encode_jpeg_ffmpeg(image, dest, ffmpeg_exe)
                except AnalyzeServeCancelled:
                    raise
                except AnalyzeServeError:
                    raise
                except Exception as exc:
                    raise _fail(
                        "review",
                        f"could not write review image {dest}: {exc}.",
                    ) from exc
                base["actual_source_time"] = actual
                base["image"] = rel
                base["status"] = "rendered"
                if anchor2_flag:
                    assert manual_by_stage is not None
                    manual_requested = manual_by_stage.get(stage)
                    base["manual_requested_source_time"] = manual_requested
                    base["manual_actual_source_time"] = None
                    base["manual_image"] = None
                    if requested is not None and manual_requested is not None:
                        base["delta_selected_minus_manual_ms"] = (
                            float(requested) - float(manual_requested)
                        ) * 1000.0
                    else:
                        base["delta_selected_minus_manual_ms"] = None
                    if manual_requested is not None:
                        manual_f = float(manual_requested)
                        if manual_f not in manual_frames_by_time:
                            raise _fail(
                                "review",
                                f"no sampled manual frame for {stage} at "
                                f"t={manual_f!r}; refusing to publish "
                                "mislinked review images.",
                            )
                        manual_actual = manual_actual_by_requested[manual_f]
                        manual_image = manual_frames_by_time[manual_f]
                        manual_rel = f"manual-{_safe_component(stage)}.jpg"
                        manual_dest = staging / manual_rel
                        try:
                            if encode_jpeg_fn is not None:
                                encode_jpeg_fn(manual_image, manual_dest)
                            else:
                                _encode_jpeg_ffmpeg(
                                    manual_image, manual_dest, ffmpeg_exe
                                )
                        except AnalyzeServeCancelled:
                            raise
                        except AnalyzeServeError:
                            raise
                        except Exception as exc:
                            raise _fail(
                                "review",
                                f"could not write review image "
                                f"{manual_dest}: {exc}.",
                            ) from exc
                        base["manual_actual_source_time"] = manual_actual
                        base["manual_image"] = manual_rel
                        image_count += 1
                entries.append(base)
                image_count += 1

            review_manifest = {
                "attempt_id": ANALYZE_SERVE_ATTEMPT_ID,
                "attempt_range": attempt_range.to_dict(),
                "entries": entries,
                "method_version": ANALYZE_SERVE_METHOD_VERSION,
                "orchestration_version": ANALYZE_SERVE_VERSION,
                "requested_range": requested_range.to_dict(),
                "schema_version": ANALYZE_SERVE_SCHEMA_VERSION,
                "source_duration_seconds": duration,
                "source_fingerprint": fingerprint,
            }
            if anchor2_flag:
                review_manifest["anchor2comparison"] = True
                review_manifest["anchor2_source"] = str(
                    ANCHOR2_ANNOTATION_RELPATH
                )
            review_text = json.dumps(review_manifest, sort_keys=True, indent=2) + "\n"
            range_text = (
                f"[{attempt_range.start_seconds}, {attempt_range.end_seconds})"
            )
            html_text = _build_index_html(
                entries,
                fingerprint=fingerprint,
                attempt_id=ANALYZE_SERVE_ATTEMPT_ID,
                range_text=range_text,
                anchor2comparison=anchor2_flag,
            )
            try:
                (staging / REVIEW_JSON_FILENAME).write_text(
                    review_text, encoding="utf-8"
                )
                (staging / INDEX_HTML_FILENAME).write_text(
                    html_text, encoding="utf-8"
                )
            except OSError as exc:
                raise _fail(
                    "review",
                    f"could not stage review index files: {exc}.",
                ) from exc
            _progress("analyze-serve: publishing")
            if review_dir.exists():
                if not force_flag:
                    raise _fail(
                        "review",
                        f"output collision: {review_dir} already exists. "
                        "Pass --force to replace the review directory. ",
                    )
                try:
                    shutil.rmtree(review_dir)
                except OSError as exc:
                    raise _fail(
                        "review",
                        f"could not replace review directory {review_dir}: "
                        f"{exc}.",
                    ) from exc
            try:
                os.replace(staging, review_dir)
            except OSError as exc:
                raise _fail(
                    "review",
                    f"could not publish review directory {review_dir}: {exc}.",
                ) from exc
        except AnalyzeServeCancelled:
            _remove_staging()
            raise
        except AnalyzeServeError:
            _remove_staging()
            raise
        except KeyboardInterrupt as exc:
            _remove_staging()
            raise AnalyzeServeCancelled(
                "review", "analyze-serve was interrupted."
            ) from exc
        except Exception as exc:
            _remove_staging()
            raise _fail("review", f"review rendering failed: {exc}.") from exc

        # --- atomic checkpoints + diagnostics (no partial final file) ---
        _progress("analyze-serve: writing checkpoints")
        if _cancelled():
            raise AnalyzeServeCancelled(
                "write",
                "analyze-serve was cancelled before writing checkpoints.",
            )
        try:
            _write_text_atomic(
                checkpoints_out, phase_document.to_json(), "write"
            )
            _write_text_atomic(diagnostics_out, diagnostics_text, "write")
            _write_text_atomic(fingerprint_out, fingerprint_text, "write")
        except AnalyzeServeError as exc:
            raise _fail("write", str(exc.message)) from exc
        except Exception as exc:
            raise _fail(
                "write", f"could not write analysis outputs: {exc}."
            ) from exc

        _progress("analyze-serve: done")
        selected_times = {
            stage: time_by_stage.get(stage) for stage in STAGE_ORDER
        }
        return AnalyzeServeResult(
            video=video_path,
            session_dir=session_dir,
            source_metadata=metadata,
            requested_range=requested_range,
            attempt_range=attempt_range,
            attempt_phase=attempt_phase,
            checkpoints_path=checkpoints_out,
            diagnostics_path=diagnostics_out,
            fingerprint_path=fingerprint_out,
            review_dir=review_dir,
            index_html=review_dir / INDEX_HTML_FILENAME,
            review_json=review_dir / REVIEW_JSON_FILENAME,
            cache_path=cache_target,
            cache_hit=cache_hit,
            racketvision_cache_path=racketvision_cache_target,
            racketvision_cache_hit=racketvision_hit,
            frame_count=len(observations),
            inferred_frames=inferred,
            total_score=float(solution.total_score),
            selected_times=selected_times,
            image_count=image_count,
            entry_count=len(entries),
            empty=False,
        )
    except AnalyzeServeCancelled as exc:
        for stem in (CHECKPOINTS_FILENAME, DIAGNOSTICS_FILENAME, FINGERPRINT_FILENAME):
            _remove_tmp_siblings(session_dir, stem)
        raise
    except AnalyzeServeError:
        for stem in (CHECKPOINTS_FILENAME, DIAGNOSTICS_FILENAME, FINGERPRINT_FILENAME):
            _remove_tmp_siblings(session_dir, stem)
        raise
