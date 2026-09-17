from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from serve_review.media.frames import SampledFrame, build_rawvideo_decode_args
from serve_review.tracking import model_frames
from serve_review.tracking.model_frames import (
    RACKETVISION_HLG_FILTER,
    ColorMetadata,
    build_color_probe_args,
    iter_racketvision_model_frames,
    parse_color_metadata,
)


def test_hlg_metadata_selects_bt709_filter() -> None:
    metadata = parse_color_metadata(
        {
            "streams": [
                {
                    "color_space": "bt2020nc",
                    "color_transfer": "arib-std-b67",
                    "color_primaries": "bt2020",
                    "color_range": "tv",
                }
            ]
        }
    )

    assert metadata.is_bt2020_hlg is True
    assert metadata.racketvision_filter == RACKETVISION_HLG_FILTER


@pytest.mark.parametrize(
    "stream",
    [
        {},
        {
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        },
    ],
)
def test_sdr_or_unspecified_metadata_does_not_transform(stream: dict) -> None:
    metadata = parse_color_metadata({"streams": [stream]})

    assert metadata.is_bt2020_hlg is False
    assert metadata.racketvision_filter is None


def test_color_probe_command_selects_first_video_stream() -> None:
    assert build_color_probe_args("serve.mov", ffprobe="custom-ffprobe") == [
        "custom-ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_transfer,color_primaries,color_range",
        "-of",
        "json",
        "serve.mov",
    ]


def test_decode_filter_composes_before_optional_fps_filter() -> None:
    args = build_rawvideo_decode_args(
        "serve.mov", rate_hz=30.0, filter_expression=RACKETVISION_HLG_FILTER
    )

    index = args.index("-vf")
    assert args[index + 1] == f"{RACKETVISION_HLG_FILTER},fps=30:round=up"


def test_model_frame_helper_preserves_sampler_pts_and_selects_hlg_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        SampledFrame(1.0, 1000, 2, 1, np.zeros((1, 2, 3), dtype=np.uint8)),
        SampledFrame(1.25, 1250, 2, 1, np.ones((1, 2, 3), dtype=np.uint8)),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        model_frames,
        "probe_color_metadata",
        lambda video, *, ffprobe: ColorMetadata(
            "bt2020nc", "arib-std-b67", "bt2020", "tv"
        ),
    )

    def fake_iter(video: object, times: object, **kwargs: object):
        captured.update(video=video, times=times, **kwargs)
        return iter(expected)

    monkeypatch.setattr(model_frames, "iter_sampled_frames", fake_iter)

    actual = tuple(
        iter_racketvision_model_frames(
            Path("serve.mov"), (1.0, 1.25), source="source-metadata"
        )
    )

    assert tuple(frame.time_seconds for frame in actual) == (1.0, 1.25)
    assert tuple(frame.timestamp_ms for frame in actual) == (1000, 1250)
    assert captured["times"] == (1.0, 1.25)
    assert captured["filter_expression"] == RACKETVISION_HLG_FILTER


def test_model_frame_helper_passes_no_filter_for_sdr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        model_frames,
        "probe_color_metadata",
        lambda video, *, ffprobe: ColorMetadata("bt709", "bt709", "bt709", "tv"),
    )

    def fake_iter(video: object, times: object, **kwargs: object):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(model_frames, "iter_sampled_frames", fake_iter)

    assert tuple(iter_racketvision_model_frames("serve.mov", (1.0,))) == ()
    assert captured["filter_expression"] is None
