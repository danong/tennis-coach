from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bootstrap_runtime


class Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        content, self.content = self.content[:size], self.content[size:]
        return content


def test_download_url_uses_huggingface_file_route() -> None:
    assert bootstrap_runtime._download_url(
        "https://huggingface.co/example/model/blob/main/checkpoint.pth"
    ) == "https://huggingface.co/example/model/resolve/main/checkpoint.pth"


def test_download_verifies_before_replacing_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.bin"
    target.write_bytes(b"old model")
    content = b"verified model bytes"
    monkeypatch.setattr(
        bootstrap_runtime.urllib.request,
        "urlopen",
        lambda request, timeout: Response(content),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        bootstrap_runtime._download_and_verify(
            "https://example.invalid/model.bin", target, "0" * 64
        )

    assert target.read_bytes() == b"old model"
    assert list(tmp_path.iterdir()) == [target]


def test_download_keeps_file_when_checksum_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.bin"
    content = b"verified model bytes"
    monkeypatch.setattr(
        bootstrap_runtime.urllib.request,
        "urlopen",
        lambda request, timeout: Response(content),
    )

    bootstrap_runtime._download_and_verify(
        "https://example.invalid/model.bin",
        target,
        hashlib.sha256(content).hexdigest(),
    )

    assert target.read_bytes() == content
