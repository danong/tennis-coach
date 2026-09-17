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
from serve_review.tracking.model_frames import (
    RACKETVISION_HLG_FILTER,
    ColorMetadata,
    build_color_probe_args,
    iter_racketvision_model_frames,
    parse_color_metadata,
    probe_color_metadata,
)
from serve_review.tracking.racketvision import (
    RacketVisionConfig,
    RacketVisionError,
    RacketVisionFrameObservation,
    RacketVisionTracker,
)

__all__ = [
    "RACKETVISION_CACHE_SCHEMA_VERSION",
    "RACKETVISION_HLG_FILTER",
    "RACKETVISION_PREPROCESSING_VERSION",
    "ColorMetadata",
    "RacketVisionCacheCorruptError",
    "RacketVisionCacheError",
    "RacketVisionCacheIdentity",
    "RacketVisionCacheSnapshot",
    "RacketVisionCacheStaleError",
    "RacketVisionConfig",
    "RacketVisionError",
    "RacketVisionFrameObservation",
    "RacketVisionTracker",
    "build_color_probe_args",
    "fingerprint_frame_times",
    "iter_racketvision_model_frames",
    "load_racketvision_cache",
    "parse_color_metadata",
    "probe_color_metadata",
    "write_racketvision_cache",
]
