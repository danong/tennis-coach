"""Focused, side-effect-free contract tests for target planning."""

from pathlib import Path

import pytest

from serve_review.domain import MediaRange, SourceMetadata
from serve_review.workflow.planning import (
    PlanningError,
    build_process_plan,
    discover_targets,
)
from serve_review.workflow.records import (
    ProcessPlan,
    SourceRecord,
    WorkflowRecordError,
    source_id_for_fingerprint,
)


def fingerprint(digit: str) -> str:
    return "sha256:" + digit * 64


def registered_source(value: str, path: Path) -> SourceRecord:
    metadata = SourceMetadata(
        fingerprint=value,
        duration_seconds=1,
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


def test_discovery_is_sorted_case_insensitive_and_reports_entries(
    tmp_path: Path,
) -> None:
    (tmp_path / "z.MP4").write_bytes(b"z")
    (tmp_path / "a.MoV").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "child").mkdir()
    files, unsupported = discover_targets(tmp_path)
    assert files == (str(tmp_path / "a.MoV"), str(tmp_path / "z.MP4"))
    assert unsupported == (
        str(tmp_path / "child"),
        str(tmp_path / "notes.txt"),
    )


def test_recursive_discovery_traverses_but_does_not_report_directories(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "child" / "deeper"
    nested.mkdir(parents=True)
    video = nested / "clip.MOV"
    video.write_bytes(b"video")
    unsupported_file = nested / "notes.txt"
    unsupported_file.write_text("notes")
    files, unsupported = discover_targets(tmp_path, recursive=True)
    assert files == (str(video),)
    assert unsupported == (str(unsupported_file),)


@pytest.mark.parametrize("name", ["missing.mov", "bad.txt"])
def test_explicit_bad_target_errors(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    if name == "bad.txt":
        target.write_text("x")
    with pytest.raises(PlanningError, match="does not exist|unsupported"):
        discover_targets(target)


def test_dedup_uses_complete_fingerprint_and_stable_path_order(tmp_path: Path) -> None:
    one, two = tmp_path / "b.mov", tmp_path / "a.mp4"
    one.write_bytes(b"same")
    two.write_bytes(b"same")
    value = fingerprint("a")
    calls: list[str] = []

    def identify(path: str) -> str:
        calls.append(path)
        return value.upper().replace("SHA256", "sha256")

    plan = build_process_plan(
        [one, two],
        fingerprint_fn=identify,
        source_lookup=lambda _: None,
    )
    assert len(plan.new) == len(plan.discovered) == 1
    assert plan.new[0].source_fingerprint == value
    assert plan.new[0].known_paths == tuple(sorted((str(one), str(two))))
    assert calls == sorted((str(one), str(two)))
    assert all(
        getattr(plan.sources[0], name).state == "required"
        for name in ("probe", "detection", "checkpoint_analysis", "landing_page")
    )


def test_registered_state_is_injected_and_missing_evidence_is_unknown(
    tmp_path: Path,
) -> None:
    video = tmp_path / "x.mov"
    video.write_bytes(b"x")
    value = fingerprint("b")
    record = registered_source(value, video)
    plan = build_process_plan(
        video,
        fingerprint_fn=lambda _: value,
        source_lookup=lambda _: record,
        state_inspector=lambda _: {
            "probe": "reusable",
            "detection": "required",
        },
    )
    source = plan.sources[0]
    assert plan.registered == (record,)
    assert [
        source.probe.state,
        source.detection.state,
        source.checkpoint_analysis.state,
        source.landing_page.state,
    ] == ["reusable", "required", "unknown", "unknown"]


def test_explicit_modes_require_exactly_one_source(tmp_path: Path) -> None:
    first = tmp_path / "x.mov"
    second = tmp_path / "y.mp4"
    first.write_bytes(b"x")
    second.write_bytes(b"y")

    def identify(path: str) -> str:
        return fingerprint("c" if path.endswith("x.mov") else "d")

    kwargs = {"fingerprint_fn": identify, "source_lookup": lambda _: None}
    plan = build_process_plan(
        first, explicit_range=MediaRange(1, 2), **kwargs
    )
    assert plan.mode == "explicit-range"
    assert plan.explicit_range == MediaRange(1, 2)
    with pytest.raises(PlanningError, match="mutually exclusive"):
        build_process_plan(
            first,
            single_attempt=True,
            explicit_range=MediaRange(1, 2),
            **kwargs,
        )
    with pytest.raises(PlanningError, match="exactly one"):
        build_process_plan([first, second], single_attempt=True, **kwargs)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PlanningError, match="exactly one"):
        build_process_plan(empty, single_attempt=True, **kwargs)


def test_collaborator_failures_and_invalid_results_are_actionable(
    tmp_path: Path,
) -> None:
    video = tmp_path / "x.mov"
    video.write_bytes(b"x")

    with pytest.raises(PlanningError, match="could not fingerprint"):
        build_process_plan(
            video,
            fingerprint_fn=lambda _: (_ for _ in ()).throw(OSError("denied")),
            source_lookup=lambda _: None,
        )
    with pytest.raises(PlanningError, match="fingerprint.*invalid"):
        build_process_plan(
            video,
            fingerprint_fn=lambda _: "not-a-fingerprint",
            source_lookup=lambda _: None,
        )
    with pytest.raises(PlanningError, match="could not look up"):
        build_process_plan(
            video,
            fingerprint_fn=lambda _: fingerprint("e"),
            source_lookup=lambda _: (_ for _ in ()).throw(OSError("broken")),
        )

    record = registered_source(fingerprint("e"), video)
    with pytest.raises(PlanningError, match="could not inspect"):
        build_process_plan(
            video,
            fingerprint_fn=lambda _: fingerprint("e"),
            source_lookup=lambda _: record,
            state_inspector=lambda _: (_ for _ in ()).throw(OSError("broken")),
        )
    with pytest.raises(PlanningError, match="unknown steps"):
        build_process_plan(
            video,
            fingerprint_fn=lambda _: fingerprint("e"),
            source_lookup=lambda _: record,
            state_inspector=lambda _: {"cache_guess": "reusable"},
        )


def test_process_plan_codec_and_human_output_are_strict_and_stable(
    tmp_path: Path,
) -> None:
    video = tmp_path / "x.mov"
    video.write_bytes(b"x")
    plan = build_process_plan(
        video,
        fingerprint_fn=lambda _: fingerprint("f"),
        source_lookup=lambda _: None,
    )
    assert ProcessPlan.from_json(plan.to_json()) == plan
    assert plan.human() == plan.human()
    assert "1 discovered, 1 new" in plan.human()

    payload = plan.to_dict()
    with pytest.raises(WorkflowRecordError):
        ProcessPlan.from_dict({**payload, "schema_version": True})
    with pytest.raises(WorkflowRecordError):
        ProcessPlan.from_dict({**payload, "surprise": True})
    with pytest.raises(WorkflowRecordError):
        ProcessPlan.from_json('{"schema_version":1,"schema_version":1}')

    two_sources = tuple(plan.sources) * 2
    with pytest.raises(WorkflowRecordError, match="unique fingerprints|exactly one"):
        ProcessPlan(1, "single-attempt", None, two_sources, ())


def test_planning_does_not_write_or_invoke_processing(tmp_path: Path) -> None:
    video = tmp_path / "x.mov"
    video.write_bytes(b"original")
    before = set(tmp_path.rglob("*"))
    build_process_plan(
        video,
        fingerprint_fn=lambda _: fingerprint("1"),
        source_lookup=lambda _: None,
    )
    assert video.read_bytes() == b"original"
    assert set(tmp_path.rglob("*")) == before
