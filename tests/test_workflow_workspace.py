from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.workflow import (
    WorkspaceError,
    WorkspacePaths,
    ensure_workspace,
    read_json,
    resolve_workspace,
    write_json_atomic,
)


def test_workspace_resolution_precedence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    env = {
        "SERVE_REVIEW_WORKSPACE": str(tmp_path / "environment"),
        "XDG_DATA_HOME": str(tmp_path / "xdg"),
    }
    assert resolve_workspace(tmp_path / "explicit", env, home).root == tmp_path / "explicit"
    assert resolve_workspace(None, env, home).root == tmp_path / "environment"
    assert resolve_workspace(None, {"XDG_DATA_HOME": str(tmp_path / "xdg")}, home).root == tmp_path / "xdg" / "serve-review"
    assert resolve_workspace(None, {}, home).root == home / ".local" / "share" / "serve-review"


@pytest.mark.parametrize(
    ("explicit", "environment", "label"),
    [
        ("", {}, "explicit workspace"),
        (None, {"SERVE_REVIEW_WORKSPACE": ""}, "SERVE_REVIEW_WORKSPACE"),
        (None, {"XDG_DATA_HOME": ""}, "XDG_DATA_HOME"),
    ],
)
def test_workspace_resolution_rejects_empty_values(
    explicit: object | None,
    environment: dict[str, object],
    label: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(WorkspaceError, match=label):
        resolve_workspace(explicit, environment, tmp_path)


def test_workspace_resolution_expands_user_without_side_effects(tmp_path: Path) -> None:
    resolved = resolve_workspace("~/serve-review-test", {}, tmp_path)
    assert not str(resolved.root).startswith("~")
    untouched = resolve_workspace(tmp_path / "missing", {}, tmp_path)
    assert not untouched.root.exists()


@pytest.mark.parametrize("value", [b"workspace", object()])
def test_workspace_resolution_wraps_unsupported_paths(
    value: object, tmp_path: Path
) -> None:
    with pytest.raises(WorkspaceError, match="explicit workspace"):
        resolve_workspace(value, {}, tmp_path)


def test_workspace_paths_are_root_derived_and_frozen(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "root")
    assert workspace.managed_directories == (
        workspace.root,
        workspace.root / "sources",
        workspace.root / "sessions",
        workspace.root / "collections",
        workspace.root / "runs",
        workspace.root / "temporary",
        workspace.root / "trash",
    )
    with pytest.raises((AttributeError, TypeError)):
        workspace.sources = tmp_path / "elsewhere"  # type: ignore[misc]
    with pytest.raises(TypeError):
        WorkspacePaths(tmp_path / "root", tmp_path / "elsewhere")  # type: ignore[call-arg]


def test_ensure_workspace_creates_exact_children_and_is_idempotent(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    assert ensure_workspace(workspace) is workspace
    assert ensure_workspace(workspace) is workspace
    assert sorted(path.name for path in workspace.root.iterdir()) == [
        "collections",
        "runs",
        "sessions",
        "sources",
        "temporary",
        "trash",
    ]
    with pytest.raises(WorkspaceError):
        ensure_workspace(tmp_path / "not-workspace")  # type: ignore[arg-type]


def test_json_is_deterministic_utf8_and_replaces_existing(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "manifest.json"
    data = {"z": "é", "a": [1, True]}
    assert write_json_atomic(destination, data) == destination
    expected = '{\n  "a": [\n    1,\n    true\n  ],\n  "z": "é"\n}\n'.encode()
    assert destination.read_bytes() == expected
    write_json_atomic(destination, {"new": 2})
    assert read_json(destination) == {"new": 2}
    assert not list(destination.parent.glob("manifest.json.tmp-*"))


@pytest.mark.parametrize("content", ["", "{bad", "[]", "null", "1", "NaN"])
def test_read_json_rejects_invalid_or_non_object_documents(
    tmp_path: Path, content: str
) -> None:
    source = tmp_path / "manifest.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(WorkspaceError, match="manifest.json"):
        read_json(source)


def test_read_json_rejects_invalid_utf8_and_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_bytes(b"{\xff")
    with pytest.raises(WorkspaceError, match="manifest.json"):
        read_json(source)
    with pytest.raises(WorkspaceError, match="missing.json"):
        read_json(tmp_path / "missing.json")


def test_write_failure_removes_actual_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serve_review.workflow.workspace as module

    real_named_temporary = module.tempfile.NamedTemporaryFile

    class FailingHandle:
        def __init__(self, handle: object) -> None:
            self._handle = handle
            self.name = handle.name  # type: ignore[attr-defined]

        def __enter__(self) -> FailingHandle:
            self._handle.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self._handle.__exit__(*args)  # type: ignore[attr-defined]

        def write(self, text: str) -> int:
            self._handle.write(text)  # type: ignore[attr-defined]
            raise OSError("write failed")

        def flush(self) -> None:
            self._handle.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            return self._handle.fileno()  # type: ignore[attr-defined,no-any-return]

    def failing_named_temporary(*args: object, **kwargs: object) -> FailingHandle:
        return FailingHandle(real_named_temporary(*args, **kwargs))

    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", failing_named_temporary)
    destination = tmp_path / "manifest.json"
    with pytest.raises(WorkspaceError, match="manifest.json"):
        write_json_atomic(destination, {"ok": True})
    assert not list(tmp_path.glob("manifest.json.tmp-*"))


def test_replace_failure_removes_unique_sibling_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serve_review.workflow.workspace as module

    observed: list[Path] = []

    def fail_replace(source: object, destination: object) -> None:
        observed.append(Path(source))
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    destination = tmp_path / "manifest.json"
    with pytest.raises(WorkspaceError, match="manifest.json"):
        write_json_atomic(destination, {"ok": True})
    assert len(observed) == 1
    assert observed[0].parent == destination.parent
    assert observed[0].name.startswith("manifest.json.tmp-")
    assert observed[0].name != "manifest.json.tmp-"
    assert not list(tmp_path.glob("manifest.json.tmp-*"))
