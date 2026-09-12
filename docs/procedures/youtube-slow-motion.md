# Proposed YouTube slow-motion corpus cookbook

This is a simple proposed workflow for adding publicly downloaded tennis videos
to a local corpus. It keeps only segments that appear to be slow motion and
leaves real-time and speed-ramp footage out of the first corpus version.

## 1. Download the best available video

Keep the untouched download outside the pipeline output tree:

```sh
mkdir -p refs/corpus/incoming
yt-dlp -f 'bv*+ba/b' --merge-output-format mp4 \
  --write-info-json \
  -o 'refs/corpus/incoming/%(id)s.%(ext)s' URL
```

Record the URL/video ID and keep the `.info.json` beside the download. Do not
modify the downloaded file.

## 2. Check the delivered format

Probe the actual download rather than trusting the source video's title:

```sh
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames,duration \
  -of json refs/corpus/incoming/VIDEO.mp4
```

The download may be 30 or 60 fps even when the original camera recording was
120 fps. That is expected; missing frames cannot be recovered.

## 3. Manually identify slow-motion intervals

Watch the video and record only intervals that are clearly slow motion. Exclude
transitions, normal-speed footage, and speed ramps. Use source presentation
times, not frame numbers divided by an assumed frame rate.

A minimal segment manifest can look like this:

```json
{
  "schema_version": 1,
  "source": "refs/corpus/incoming/VIDEO.mp4",
  "segments": [
    {
      "id": "serve-001",
      "start_seconds": 42.18,
      "end_seconds": 46.92,
      "label": "serve",
      "slow_motion": true,
      "capture_fps_assumption": 120,
      "delivery_fps": 30,
      "slowdown_factor": 4
    }
  ]
}
```

The 120-fps value is a corpus assumption, not a claim that the download still
contains 120 unique frames per second.

## 4. Sanity-check the 120-fps assumption

Preview a candidate segment at approximately 4× playback speed:

```sh
ffplay -vf 'setpts=0.25*PTS' -an refs/corpus/incoming/VIDEO.mp4
```

Or use the equivalent speed control in a video player. If motion looks close
to real-time at 4×, retain the segment as a provisional 120-fps slow-motion
sample. If it remains obviously slow, fast, or changes speed, reject it from
this first-pass corpus.

This is an eyeball quality check, not a recovery of the original capture rate.

## 5. Export the retained segment

Create separate working derivatives from the untouched download with the
repository helper:

```sh
uv run python tools/export_segments.py \
  refs/corpus/incoming/VIDEO.mp4 \
  refs/corpus/VIDEO.segments.json \
  --output-dir refs/corpus/segments/VIDEO
```

The helper validates IDs and source-time bounds, re-encodes with FFmpeg, and
publishes each clip atomically. It refuses existing outputs unless
`--overwrite` is supplied. The original download remains the provenance
source. Do not interpolate frames or convert the segment to nominal 120 fps by
repeating frames.

## 6. Ingest the segment

Run the normal serve-cut workflow on each retained segment or on a manually
assembled session, depending on the corpus experiment:

```sh
mise run cut -- refs/corpus/segments/VIDEO/serve-001.mp4 \
  --padding 0 \
  --output clips \
  --output-dir output/youtube-corpus/VIDEO
```

For a hand-selected single serve, the exported segment itself can be copied
to an ignored local anchor/corpus location using the project’s normal naming
convention.

## 7. Keep the metadata simple

For each retained segment, preserve:

- source URL/video ID;
- download filename and checksum;
- delivered codec, resolution, and frame rate;
- segment start/end in source presentation time;
- `capture_fps_assumption` (usually 120 for this corpus);
- `delivery_fps` (the measured downloaded rate);
- `slowdown_factor` (usually 4 for 120-to-30 delivery);
- manual review status and reviewer notes.

## Scope and limitations

- This corpus is for exploratory detection and phase work, not strict native
  120-fps timing benchmarks.
- The delivered video is the authoritative timeline for pipeline annotations.
- Do not use frame interpolation as ground truth.
- Keep real-time and ramped clips out until the corpus has an explicit model for
  variable playback speed.
- Follow applicable copyright, licensing, and platform terms when downloading
  and storing source material.
