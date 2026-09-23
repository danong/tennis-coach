#!/usr/bin/env python3
"""Fetch the ignored RacketVision checkout and model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR_URL = "https://github.com/OrcustD/RacketVision.git"
VENDOR_COMMIT = "c44af2a08524d3cb54d818f19686f4cdea4d2793"
MANIFEST = ROOT / "models" / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vendor_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_vendor(destination: Path = ROOT / "vendor" / "racketvision") -> str:
    if destination.exists():
        try:
            revision = _vendor_revision(destination)
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(
                f"{destination} exists but is not a readable Git checkout; "
                "move it aside and rerun bootstrap"
            ) from error
        if revision != VENDOR_COMMIT:
            raise RuntimeError(
                f"{destination} is at {revision}, expected {VENDOR_COMMIT}; "
                "move it aside and rerun bootstrap"
            )
        return "already present"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="racketvision-", dir=destination.parent) as temp:
        checkout = Path(temp) / "checkout"
        subprocess.run(["git", "clone", VENDOR_URL, str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", VENDOR_COMMIT],
            check=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkout.rename(destination)
    return "cloned"


def _download_url(source_url: str) -> str:
    # Hugging Face's /blob/ route serves a viewer page; /resolve/ serves bytes.
    return source_url.replace("/blob/", "/resolve/", 1)


def _download_and_verify(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "serve-review-bootstrap/1"})
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{target.name}.", suffix=".download", dir=target.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {target}: expected {expected_sha256}, got {actual_sha256}"
        )
    os.replace(temporary_path, target)


def ensure_models(manifest_path: Path = MANIFEST, root: Path = ROOT) -> list[tuple[str, str]]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), list):
        raise RuntimeError(f"Unsupported model manifest: {manifest_path}")

    outcomes: list[tuple[str, str]] = []
    for model in manifest["models"]:
        artifact = model["artifact"]
        target = root / "models" / artifact
        expected = model["sha256"]
        if target.is_file() and _sha256(target) == expected:
            outcomes.append((artifact, "verified"))
            continue
        _download_and_verify(_download_url(model["source_url"]), target, expected)
        outcomes.append((artifact, "downloaded and verified"))
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="project root (primarily useful for local checks)"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        vendor_state = ensure_vendor(root / "vendor" / "racketvision")
        print(f"RacketVision: {vendor_state} at {VENDOR_COMMIT}")
        for artifact, state in ensure_models(root / "models" / "manifest.json", root):
            print(f"{artifact}: {state}")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"bootstrap failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
