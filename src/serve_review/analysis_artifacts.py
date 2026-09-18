"""Description and validation of one analysis attempt's generated artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.analyze_serve import (
    CHECKPOINTS_FILENAME,
    DIAGNOSTICS_FILENAME,
    INDEX_HTML_FILENAME,
    RACKETVISION_CACHE_FILENAME,
    REVIEW_DIRNAME,
    REVIEW_JSON_FILENAME,
)


@dataclass(frozen=True, slots=True)
class AttemptAnalysisArtifacts:
    """Paths and completeness validation for one analyzed attempt.

    This object describes existing analysis output only. Workflow coordinators
    retain ownership of clearing destinations and scheduling reruns.
    """

    directory: Path

    @property
    def checkpoints_path(self) -> Path:
        return self.directory / CHECKPOINTS_FILENAME

    @property
    def diagnostics_path(self) -> Path:
        return self.directory / DIAGNOSTICS_FILENAME

    @property
    def review_dir(self) -> Path:
        return self.directory / REVIEW_DIRNAME

    @property
    def index_html_path(self) -> Path:
        return self.review_dir / INDEX_HTML_FILENAME

    @property
    def racketvision_cache_path(self) -> Path:
        return self.directory / "cache" / RACKETVISION_CACHE_FILENAME

    def is_complete(self) -> bool:
        if not self.racketvision_cache_path.is_file():
            return False
        checkpoints = _json_object(self.checkpoints_path)
        diagnostics = _json_object(self.diagnostics_path)
        review = _json_object(self.review_dir / REVIEW_JSON_FILENAME)
        try:
            html = self.index_html_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False
        if checkpoints is None or diagnostics is None or review is None or not html.strip():
            return False
        entries = review.get("entries")
        if not isinstance(entries, list):
            return False
        for entry in entries:
            if not isinstance(entry, dict):
                return False
            for key in ("image", "manual_image"):
                image = entry.get(key)
                if image and not (self.review_dir / Path(str(image)).name).is_file():
                    return False
        return True


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None
