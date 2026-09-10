"""Pose inference backend contract and shared helpers (M2.4).

This module is runtime-agnostic: it owns the backend protocol, the error
hierarchy, canonical-timestamp helpers, and model-file verification. It
never imports MediaPipe (or any model runtime), never touches private
footage, and performs no network access. MediaPipe-specific mapping lives
in :mod:`serve_review.pose.mediapipe`.

Timestamp contract (shared with the M2.1 sampler and M2.2 schemas):

- ``time_seconds`` (float) is canonical source time and is preserved
  exactly on every :class:`~serve_review.pose.schema.FrameObservation`.
- ``timestamp_ms`` (int) is only the ``round(time_seconds * 1000)``
  convenience required for ordered MediaPipe ``VIDEO`` calls. Canonical
  logic must never derive time from a frame index or an assumed FPS.
- ``VIDEO`` calls require strictly increasing integer milliseconds, so
  adapters must track the last submitted value and reject repeats or
  steps backwards with an actionable error instead of silently coercing.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "PoseBackendError",
    "ModelFileError",
    "ModelHashMismatchError",
    "TimestampOrderError",
    "InferenceError",
    "BackendClosedError",
    "MappingError",
    "PoseBackend",
    "timestamp_ms_for",
    "check_timestamp_order",
    "sha256_of_file",
    "verify_model_file",
]


class PoseBackendError(Exception):
    """Base error for pose backend failures."""


class ModelFileError(PoseBackendError):
    """Raised when the approved model artifact cannot be used."""


class ModelHashMismatchError(ModelFileError):
    """Raised when a model file fails SHA-256 verification."""


class TimestampOrderError(PoseBackendError):
    """Raised when VIDEO timestamps are not strictly increasing."""


class InferenceError(PoseBackendError):
    """Raised when the underlying estimator call fails."""


class BackendClosedError(PoseBackendError):
    """Raised when inference is attempted on a closed backend."""


class MappingError(PoseBackendError):
    """Raised when estimator output violates the 33-landmark contract."""


class PoseBackend(Protocol):
    """Structural contract for pose inference backends.

    Implementations serialize calls to exactly one underlying estimator,
    preserve canonical ``time_seconds`` on every returned observation,
    and expose honest no-person/missing-joint output (never fabricated
    poses). This protocol exists for typing and fakes; the MediaPipe
    implementation lives in :mod:`serve_review.pose.mediapipe`.
    """

    @property
    def model_name(self) -> str:
        """Return the estimator identity for cache headers."""
        ...

    @property
    def model_version(self) -> str:
        """Return the estimator version for cache headers."""
        ...

    def infer(self, image: Any, time_seconds: float) -> Any:
        """Run one ordered inference and return a FrameObservation."""
        ...

    def close(self) -> None:
        """Release the underlying estimator (idempotent)."""
        ...


def timestamp_ms_for(time_seconds: float) -> int:
    """Return ``round(time_seconds * 1000)`` for ordered VIDEO calls.

    Raises :class:`TimestampOrderError` for non-finite or negative input
    so bad schedules fail loudly instead of producing a bogus stamp.
    """
    if (
        isinstance(time_seconds, bool)
        or not isinstance(time_seconds, (int, float))
        or not math.isfinite(float(time_seconds))
        or float(time_seconds) < 0
    ):
        raise TimestampOrderError(
            f"invalid canonical time_seconds {time_seconds!r}: expected a "
            "finite number of seconds >= 0; VIDEO timestamps must derive "
            "from canonical source time, never from a frame index."
        )
    return int(round(float(time_seconds) * 1000))


def check_timestamp_order(last_ms: int | None, next_ms: int) -> None:
    """Require ``next_ms`` to be strictly greater than ``last_ms``.

    ``last_ms`` of ``None`` means no call has been submitted yet and
    always passes. Raises :class:`TimestampOrderError` with an actionable
    message on repeats or steps backwards.
    """
    if last_ms is None:
        return
    if (
        isinstance(next_ms, bool)
        or not isinstance(next_ms, int)
        or isinstance(last_ms, bool)
        or not isinstance(last_ms, int)
    ):
        raise TimestampOrderError(
            f"invalid VIDEO timestamps last_ms={last_ms!r}, "
            f"next_ms={next_ms!r}: both must be integers."
        )
    if not next_ms > last_ms:
        raise TimestampOrderError(
            f"MediaPipe VIDEO timestamps must be strictly increasing: "
            f"last submitted {last_ms} ms, requested {next_ms} ms. "
            "Feed frames in increasing canonical source-time order and "
            "do not reuse one landmarker across interleaved schedules."
        )


def sha256_of_file(path: Path | str) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    target = Path(path).expanduser()
    try:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ModelFileError(
            f"pose model file does not exist: {target}."
        ) from exc
    except OSError as exc:
        raise ModelFileError(
            f"could not read pose model file {target}: {exc}."
        ) from exc
    return digest.hexdigest()


def verify_model_file(
    path: Path | str, *, expected_sha256: str | None = None
) -> Path:
    """Validate that ``path`` is a usable model file.

    Checks existence and regular-file status, then -- when
    ``expected_sha256`` is given -- verifies the SHA-256 digest. A
    mismatch raises :class:`ModelHashMismatchError` and the file must
    never be used for inference. Returns the expanded path.
    """
    target = Path(path).expanduser()
    if not target.exists():
        raise ModelFileError(
            f"pose model not found at {target}; place the approved "
            "artifact there before running pose extraction (no automatic "
            "download is attempted)."
        )
    if not target.is_file():
        raise ModelFileError(
            f"pose model path is not a regular file: {target}."
        )
    if expected_sha256 is None:
        return target
    actual = sha256_of_file(target)
    if actual.lower() != expected_sha256.lower():
        raise ModelHashMismatchError(
            f"pose model at {target} failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual}. Re-fetch the "
            "approved artifact and refuse to run with an unverified model."
        )
    return target
