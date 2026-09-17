"""Optional visual-object tracking providers and raw observation caches."""

from serve_review.tracking.cache import (
    RACKETVISION_CACHE_SCHEMA_VERSION,
    RACKETVISION_PREPROCESSING_VERSION,
    RacketVisionCacheCorruptError,
    RacketVisionCacheError,
    RacketVisionCacheIdentity,
    RacketVisionCacheSnapshot,
    RacketVisionCacheStaleError,
    fingerprint_frame_times,
    load_racketvision_cache,
    write_racketvision_cache,
)
from serve_review.tracking.racketvision import (
    RacketVisionConfig,
    RacketVisionError,
    RacketVisionFrameObservation,
    RacketVisionTracker,
)

__all__ = [
    "RACKETVISION_CACHE_SCHEMA_VERSION",
    "RACKETVISION_PREPROCESSING_VERSION",
    "RacketVisionCacheCorruptError",
    "RacketVisionCacheError",
    "RacketVisionCacheIdentity",
    "RacketVisionCacheSnapshot",
    "RacketVisionCacheStaleError",
    "RacketVisionConfig",
    "RacketVisionError",
    "RacketVisionFrameObservation",
    "RacketVisionTracker",
    "fingerprint_frame_times",
    "load_racketvision_cache",
    "write_racketvision_cache",
]
