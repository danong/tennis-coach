from __future__ import annotations

import pytest

from serve_review.media.color import (
    HLG_TO_SDR_FILTER,
    build_color_probe_args,
    parse_color_metadata,
)


def test_tagged_iphone_hlg_selects_tonemapping_filter() -> None:
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
    assert metadata.sdr_filter == HLG_TO_SDR_FILTER
    assert "pin=bt2020:tin=arib-std-b67:min=bt2020nc:rin=tv" in HLG_TO_SDR_FILTER
    assert "transfer=linear" in HLG_TO_SDR_FILTER
    assert "tonemap=tonemap=hable" in HLG_TO_SDR_FILTER


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
def test_sdr_or_unknown_sources_need_no_conversion(stream: dict) -> None:
    metadata = parse_color_metadata({"streams": [stream]})

    assert metadata.is_bt2020_hlg is False
    assert metadata.sdr_filter is None


def test_color_probe_selects_the_first_video_stream() -> None:
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
