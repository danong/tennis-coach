#!/usr/bin/env python3
"""Write a deterministic M4 checkpoint evaluation report without mutating inputs."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from serve_review.checkpoint_evaluation import (
    CheckpointEvaluationError,
    PhaseAnnotationManifest,
    evaluate_phase_document,
)
from serve_review.domain import DomainError, PhaseDocument


def atomic_write(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def evaluate(checkpoints: Path, annotations: Path, output: Path, *, overwrite: bool = False) -> Path:
    try:
        document = PhaseDocument.from_json(checkpoints.read_text(encoding="utf-8"))
        manifest = PhaseAnnotationManifest.from_json(annotations.read_text(encoding="utf-8"))
        report = evaluate_phase_document(document, manifest)
    except (OSError, DomainError, CheckpointEvaluationError) as exc:
        raise ValueError(f"cannot evaluate checkpoints: {exc}") from exc
    atomic_write(output, report.to_json(), overwrite)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate phase checkpoints against manual annotations.")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        print(evaluate(args.checkpoints, args.annotations, args.output, overwrite=args.overwrite))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())
