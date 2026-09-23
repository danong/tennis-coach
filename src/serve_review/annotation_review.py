"""Local, source-bound annotation UI for one recording session."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from serve_review.process import discover, generated_paths

STAGES = (
    "start", "release", "loading", "cocking", "acceleration", "contact",
    "deceleration", "finish",
)
LABELS = {"serve", "shadow_swing", "toss_abort", "other", "ambiguous"}
STATUSES = {"accepted", "corrected", "uncertain", "unavailable"}
ANNOTATION_FILENAME = "review-annotations-v1.json"
MAX_REQUEST_BYTES = 2_000_000


class ReviewError(ValueError):
    """Invalid session data or annotation request."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"expected an object in {path}")
    return value


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and abs(value) < float("inf")


class ReviewSession:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve()
        self.videos = discover(self.directory)
        self.by_name = {video.name: video for video in self.videos}
        self.annotations_path = self.directory / "annotations" / ANNOTATION_FILENAME
        self.media_paths: dict[str, Path] = {}

    def _source(self, video: Path) -> dict:
        metadata, _ = generated_paths(video)
        return _read_json(metadata / "source.json")

    def _empty_annotations(self) -> dict:
        return {"schema_version": 1, "session_id": self.directory.name, "videos": {}}

    def annotations(self) -> dict:
        return _read_json(self.annotations_path) if self.annotations_path.exists() else self._empty_annotations()

    def prepare_media(self) -> None:
        """Remux source codecs into browser-friendly local media without re-encoding video."""
        proxy_dir = self.directory / "metadata" / "review-media"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        ready = {row["name"] for row in self.state()["videos"]}
        for video in self.videos:
            if video.name not in ready:
                continue
            source = self._source(video)
            codec = source.get("video_codec")
            if codec not in {"h264", "vp9"}:
                raise ReviewError(f"browser review does not support {video.name} codec {codec!r}")
            suffix = ".mp4" if codec == "h264" else ".webm"
            target = proxy_dir / f"{video.stem}{suffix}"
            identity = proxy_dir / f"{video.stem}.json"
            expected = {"source_fingerprint": source["fingerprint"], "codec": codec}
            if target.is_file() and identity.is_file() and _read_json(identity) == expected:
                self.media_paths[video.name] = target
                continue
            print(f"Preparing browser playback: {video.name}", flush=True)
            temporary = proxy_dir / f"{video.stem}.tmp{suffix}"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                       "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy"]
            if codec == "h264":
                command += ["-c:a", "copy", "-movflags", "+faststart"]
            else:
                command += ["-c:a", "libopus", "-b:a", "96k"]
            command.append(str(temporary))
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    raise ReviewError(f"could not prepare {video.name}: {result.stderr.strip()}")
                os.replace(temporary, target)
                identity.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
                self.media_paths[video.name] = target
            finally:
                temporary.unlink(missing_ok=True)

    def state(self) -> dict:
        annotations = self.annotations()
        rows = []
        pending = []
        for video in self.videos:
            metadata, _ = generated_paths(video)
            if not all((metadata / filename).is_file() for filename in ("source.json", "attempts.json")):
                pending.append(video.name)
                continue
            source = self._source(video)
            attempts = _read_json(metadata / "attempts.json")
            saved = annotations["videos"].get(video.name)
            if saved and saved.get("source_fingerprint") != source.get("fingerprint"):
                raise ReviewError(f"source changed since labeling: {video.name}")
            stages = {}
            for attempt in attempts.get("attempts", []):
                attempt_id = attempt["attempt_id"]
                checkpoint_path = metadata / "attempts" / attempt_id / "checkpoints.json"
                if not checkpoint_path.is_file():
                    continue
                entries = _read_json(checkpoint_path).get("attempts", [])
                # Each per-crop analysis uses its own local attempt ID. The
                # parent directory is the source-video detection identity.
                if len(entries) != 1:
                    continue
                result = entries[0]
                images = checkpoint_path.parent / "review-serve-3d"
                stages[attempt_id] = {
                    name: {
                        "keyframe_seconds": proposal.get("keyframe_seconds"),
                        "thumbnail": (images / f"{name}.jpg").is_file(),
                    }
                    for name, proposal in result.get("stages", {}).items() if name in STAGES
                }
            rows.append({
                "name": video.name,
                "stem": video.stem,
                "source_fingerprint": source["fingerprint"],
                "duration_seconds": source["duration_seconds"],
                "frame_rate": source["frame_rate_num"] / source["frame_rate_den"],
                "attempts": attempts.get("attempts", []),
                "stages": stages,
                "annotation": saved or {
                    "source_fingerprint": source["fingerprint"],
                    "view": "",
                    "fully_reviewed": False,
                    "labels": [],
                },
            })
        return {"session_id": self.directory.name, "videos": rows, "pending": pending, "stages": STAGES}

    def save_video(self, name: str, data: object) -> None:
        video = self.by_name.get(name)
        if video is None or not isinstance(data, dict):
            raise ReviewError("unknown video or invalid annotation object")
        source = self._source(video)
        if set(data) != {"source_fingerprint", "view", "fully_reviewed", "labels"}:
            raise ReviewError("annotation object has missing or unknown fields")
        if data["source_fingerprint"] != source["fingerprint"]:
            raise ReviewError("source fingerprint mismatch")
        if not isinstance(data["view"], str) or len(data["view"]) > 80:
            raise ReviewError("view must be short text")
        if type(data["fully_reviewed"]) is not bool or not isinstance(data["labels"], list):
            raise ReviewError("invalid review state or labels")
        if len(data["labels"]) > 2000:
            raise ReviewError("too many labels")
        ids = set()
        duration = source["duration_seconds"]
        for label in data["labels"]:
            if not isinstance(label, dict) or set(label) != {
                "id", "source_id", "start_seconds", "end_seconds", "label", "checkpoints"
            }:
                raise ReviewError("invalid label fields")
            identifier = label["id"]
            if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identifier) or identifier in ids:
                raise ReviewError("invalid or duplicate label id")
            ids.add(identifier)
            if label["source_id"] is not None and (
                not isinstance(label["source_id"], str)
                or not re.fullmatch(r"(?:attempt|shadow):[a-zA-Z0-9_-]{1,80}", label["source_id"])
            ):
                raise ReviewError("invalid source id")
            start, end = label["start_seconds"], label["end_seconds"]
            if not (_finite_number(start) and _finite_number(end) and 0 <= start < end <= duration + 0.001):
                raise ReviewError("invalid label range")
            if label["label"] not in LABELS or not isinstance(label["checkpoints"], dict):
                raise ReviewError("invalid label or checkpoints")
            if label["label"] != "serve" and label["checkpoints"]:
                raise ReviewError("only serves may have checkpoints")
            if set(label["checkpoints"]) - set(STAGES):
                raise ReviewError("unknown checkpoint stage")
            for value in label["checkpoints"].values():
                if not isinstance(value, dict) or set(value) != {"status", "time_seconds"}:
                    raise ReviewError("invalid checkpoint fields")
                status, moment = value["status"], value["time_seconds"]
                if status not in STATUSES or (
                    (status in {"accepted", "corrected"} and not (_finite_number(moment) and start <= moment <= end))
                    or (status in {"uncertain", "unavailable"} and moment is not None)
                ):
                    raise ReviewError("invalid checkpoint time or status")
        document = self.annotations()
        if document.get("schema_version") != 1 or document.get("session_id") != self.directory.name:
            raise ReviewError("annotation document belongs to another session")
        document["videos"][name] = data
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix="review-", suffix=".json", dir=self.annotations_path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.annotations_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], session: ReviewSession):
        self.session = session
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: object) -> None:
        if not self.path.startswith("/media/"):
            super().log_message(format, *args)

    def _json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str, *, ranged: bool = False) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range") if ranged else None
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if not match or (not match[1] and not match[2]):
                self.send_error(416)
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start >= size or end < start:
                self.send_error(416)
                return
        length = end - start + 1
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if ranged:
            self.send_header("Accept-Ranges", "bytes")
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            while length:
                block = stream.read(min(length, 1024 * 1024))
                if not block:
                    break
                self.wfile.write(block)
                length -= len(block)

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        try:
            if parts.path == "/":
                self._file(Path(__file__).with_name("annotation_review.html"), "text/html; charset=utf-8")
            elif parts.path == "/api/state":
                self._json(self.server.session.state())
            elif parts.path == "/api/refresh":
                self.server.session.prepare_media()
                self._json(self.server.session.state())
            elif parts.path.startswith("/media/"):
                name = unquote(parts.path.removeprefix("/media/"))
                video = self.server.session.by_name.get(name)
                if video is None:
                    self.send_error(404)
                else:
                    playable = self.server.session.media_paths.get(name, video)
                    content_type = "video/webm" if playable.suffix.lower() == ".webm" else "video/mp4"
                    self._file(playable, content_type, ranged=True)
            elif parts.path == "/thumb":
                query = parse_qs(parts.query)
                name = query.get("video", [""])[0]
                attempt_id = query.get("attempt", [""])[0]
                stage = query.get("stage", [""])[0]
                video = self.server.session.by_name.get(name)
                if video is None or stage not in STAGES or not re.fullmatch(r"serve-\d+", attempt_id):
                    self.send_error(404)
                    return
                metadata, _ = generated_paths(video)
                image = metadata / "attempts" / attempt_id / "review-serve-3d" / f"{stage}.jpg"
                if image.is_file():
                    self._file(image, "image/jpeg")
                else:
                    self.send_error(404)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers routinely cancel video range requests after a seek.
            return
        except (OSError, ReviewError, KeyError, ZeroDivisionError) as exc:
            self._json({"error": str(exc)}, 500)

    def do_PUT(self) -> None:
        if urlsplit(self.path).path != "/api/annotations":
            self.send_error(404)
            return
        try:
            count = int(self.headers.get("Content-Length", "0"))
            if count <= 0 or count > MAX_REQUEST_BYTES:
                raise ReviewError("invalid annotation request size")
            request = json.loads(self.rfile.read(count))
            if not isinstance(request, dict) or set(request) != {"video", "annotation"}:
                raise ReviewError("expected video and annotation")
            self.server.session.save_video(request["video"], request["annotation"])
            self._json({"saved": True})
        except (ValueError, KeyError, TypeError, ReviewError) as exc:
            self._json({"error": str(exc)}, 400)


def serve(directory: Path, port: int = 8765) -> None:
    session = ReviewSession(directory)
    if not session.state()["videos"]:
        raise ReviewError("no processed videos are ready yet; retry when a source finishes")
    session.prepare_media()
    server = ReviewServer(("127.0.0.1", port), session)
    print(f"Review {session.directory} at http://127.0.0.1:{server.server_port}/", flush=True)
    print("Press Ctrl-C to stop. Annotations stay in the recording directory.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
