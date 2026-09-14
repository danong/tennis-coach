from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.cli import build_parser
from serve_review.domain import SourceMetadata
from serve_review.workflow.cli import process_command
from serve_review.workflow.records import SourceRecord, source_id_for_fingerprint


def fingerprint(digit: str = "a") -> str:
    return "sha256:" + digit * 64


def source_record(path: Path, value: str | None = None) -> SourceRecord:
    value = value or fingerprint()
    metadata = SourceMetadata(
        fingerprint=value,
        duration_seconds=2,
        width=1,
        height=1,
        frame_rate_num=1,
        video_codec="h264",
    )
    return SourceRecord(
        1,
        source_id_for_fingerprint(value),
        value,
        metadata,
        (str(path.resolve()),),
    )


def parse(*arguments: str):
    return build_parser().parse_args(["process", *arguments])


def fake_render(page, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("page", encoding="utf-8")
    return path


def fake_batch(entries, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("index", encoding="utf-8")
    return path


def complete_result(status: str = "complete"):
    return SimpleNamespace(overall_status=status, attempts=())


def test_parser_adds_process_options_without_changing_existing_commands(
    tmp_path: Path,
) -> None:
    args = parse(
        str(tmp_path / "one.mov"),
        str(tmp_path / "two.mp4"),
        "--workspace",
        str(tmp_path / "workspace"),
        "--recursive",
        "--session",
        "Practice",
        "--single-attempt",
        "--padding",
        "0.5",
        "--open",
        "--dry-run",
    )
    assert args.command == "process"
    assert len(args.targets) == 2
    assert args.padding == 0.5
    assert args.recursive and args.single_attempt and args.open_page and args.dry_run
    assert build_parser().parse_args(["doctor"]).command == "doctor"
    assert build_parser().parse_args(["cut", "video.mov"]).command == "cut"


@pytest.mark.parametrize(
    "arguments",
    [
        ("video.mov", "--range", "bad"),
        ("video.mov", "--range", "2:1"),
        ("video.mov", "--range", "nan:2"),
        ("video.mov", "--padding", "-1"),
        ("video.mov", "--padding", "inf"),
    ],
)
def test_parser_rejects_invalid_range_and_padding(arguments) -> None:
    with pytest.raises(SystemExit) as error:
        parse(*arguments)
    assert error.value.code == 2


def test_dry_run_is_read_only_and_calls_no_mutating_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    workspace = tmp_path / "does-not-exist"

    def forbidden(*args, **kwargs):
        raise AssertionError("mutating service called during dry-run")

    exit_code = process_command(
        parse(str(video), "--workspace", str(workspace), "--dry-run"),
        services={
            "fingerprint": lambda _: fingerprint(),
            "source_lookup": lambda _: None,
            "ensure_workspace": forbidden,
            "register_source": forbidden,
            "process_source": forbidden,
            "render": forbidden,
            "render_batch": forbidden,
            "opener": forbidden,
        },
    )
    assert exit_code == 0
    assert not workspace.exists()
    output = capsys.readouterr().out
    assert f"Workspace: {workspace}" in output
    assert "Mode: normal" in output


def test_single_source_flow_prints_paths_and_opens_only_when_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    workspace = tmp_path / "workspace"
    record = source_record(video)
    process_calls = []
    opened = []

    def process(source, paths, mode, **kwargs):
        process_calls.append((source, paths, mode, kwargs))
        return complete_result()

    services = {
        "fingerprint": lambda _: record.source_fingerprint,
        "source_lookup": lambda _: None,
        "register_source": lambda *_args, **_kwargs: record,
        "process_source": process,
        "render": fake_render,
        "render_batch": fake_batch,
        "opener": lambda path: opened.append(path),
    }
    args = parse(
        str(video),
        "--workspace",
        str(workspace),
        "--range",
        "0.25:1.5",
        "--padding",
        "0.75",
    )
    assert process_command(args, services=services) == 0
    assert opened == []
    assert process_calls[0][2] == "explicit-range"
    assert process_calls[0][3]["explicit_range"].to_dict()["start_seconds"] == 0.25
    assert process_calls[0][3]["padding_seconds"] == 0.75
    output = capsys.readouterr().out
    assert "Primary landing:" in output
    assert "Recovery: serve-review status --workspace" in output

    args.open_page = True
    assert process_command(args, services=services) == 0
    assert len(opened) == 1


def test_batch_is_deterministic_registers_aliases_and_uses_temporary_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "z.mov"
    alias = tmp_path / "a.mp4"
    second = tmp_path / "m.mov"
    for path in (first, alias, second):
        path.write_bytes(b"source")
    values = {
        str(first): fingerprint("1"),
        str(alias): fingerprint("1"),
        str(second): fingerprint("2"),
    }
    records = {
        fingerprint("1"): source_record(alias, fingerprint("1")),
        fingerprint("2"): source_record(second, fingerprint("2")),
    }
    registered_paths = []
    processed = []
    batch_destinations = []

    def register(path, workspace, **kwargs):
        registered_paths.append(str(path))
        return records[values[str(path)]]

    def process(record, workspace, mode, **kwargs):
        processed.append(record.source_id)
        return complete_result()

    def batch(entries, destination):
        batch_destinations.append(Path(destination))
        return fake_batch(entries, destination)

    exit_code = process_command(
        parse(
            str(first),
            str(alias),
            str(second),
            "--workspace",
            str(tmp_path / "workspace"),
        ),
        services={
            "fingerprint": lambda path: values[path],
            "source_lookup": lambda _: None,
            "register_source": register,
            "process_source": process,
            "render": fake_render,
            "render_batch": batch,
        },
    )
    assert exit_code == 0
    assert registered_paths == [str(alias), str(first), str(second)]
    assert processed == [records[fingerprint("1")].source_id, records[fingerprint("2")].source_id]
    assert batch_destinations == [tmp_path / "workspace" / "temporary" / "index.html"]
    assert "Primary landing:" in capsys.readouterr().out


def test_named_session_is_mutated_only_after_registration(tmp_path: Path) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    record = source_record(video)
    events = []
    session = SimpleNamespace(
        session_id="session-0123456789abcdef", source_ids=()
    )

    def register(*args, **kwargs):
        events.append("register")
        return record

    def missing(*args, **kwargs):
        raise RuntimeError("not found")

    def create(*args, **kwargs):
        events.append("create-session")
        return session

    def add(*args, **kwargs):
        events.append("add-session")
        return SimpleNamespace(
            session_id=session.session_id, source_ids=(record.source_id,)
        )

    result = process_command(
        parse(
            str(video),
            "--workspace",
            str(tmp_path / "workspace"),
            "--session",
            "Practice",
        ),
        services={
            "fingerprint": lambda _: record.source_fingerprint,
            "source_lookup": lambda _: None,
            "register_source": register,
            "resolve_session": missing,
            "create_session": create,
            "add_session_source": add,
            "process_source": lambda *args, **kwargs: complete_result(),
            "render": fake_render,
        },
    )
    assert result == 0
    assert events == ["register", "create-session", "add-session"]
    assert (tmp_path / "workspace" / "sessions" / f"{session.session_id}-{record.source_id}.html").is_file()


def test_ambiguous_session_is_input_error_and_prevents_processing(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    record = source_record(video)
    processed = False

    def ambiguous(*args, **kwargs):
        raise RuntimeError("ambiguous: session-a, session-b")

    def process(*args, **kwargs):
        nonlocal processed
        processed = True

    result = process_command(
        parse(str(video), "--session", "same", "--workspace", str(tmp_path / "w")),
        services={
            "fingerprint": lambda _: record.source_fingerprint,
            "source_lookup": lambda _: None,
            "register_source": lambda *args, **kwargs: record,
            "resolve_session": ambiguous,
            "process_source": process,
        },
    )
    assert result == 2
    assert processed is False


def test_batch_exit_codes_distinguish_failure_and_partial(tmp_path: Path) -> None:
    first = tmp_path / "a.mov"
    second = tmp_path / "b.mov"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    values = {str(first): fingerprint("1"), str(second): fingerprint("2")}
    records = {value: source_record(path, value) for path, value in ((first, fingerprint("1")), (second, fingerprint("2")))}

    base_services = {
        "fingerprint": lambda path: values[path],
        "source_lookup": lambda _: None,
        "register_source": lambda path, *_args, **_kwargs: records[values[str(path)]],
        "render": fake_render,
        "render_batch": fake_batch,
    }
    args = parse(str(first), str(second), "--workspace", str(tmp_path / "w"))

    def mixed(record, *args, **kwargs):
        if record == records[fingerprint("2")]:
            raise RuntimeError("failed")
        return complete_result()

    assert process_command(args, services={**base_services, "process_source": mixed}) == 3
    assert process_command(
        args,
        services={
            **base_services,
            "process_source": lambda *args, **kwargs: complete_result("partial"),
        },
    ) == 3
    assert process_command(
        args,
        services={
            **base_services,
            "process_source": lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
        },
    ) == 1


def test_empty_directory_publishes_honest_empty_temporary_index(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    workspace = tmp_path / "workspace"
    result = process_command(
        parse(str(target), "--workspace", str(workspace)),
        services={
            "fingerprint": lambda _: pytest.fail("no source expected"),
            "source_lookup": lambda _: None,
            "render_batch": fake_batch,
        },
    )
    assert result == 0
    assert (workspace / "temporary" / "index.html").is_file()
