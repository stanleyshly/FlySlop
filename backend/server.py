"""Small local HTTP server for replay and the static web client."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mimetypes
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .corpus import get_target, list_targets
from .embodiment import replay_text, scripted_episode
from .judge import judge_text
from .keyboard import KEY_PITCH_MM, layout_payload


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FLY_ASSETS = {"/api/fly/model.json": ROOT / "data" / "neuromechfly" / "model.json",
              "/api/fly/meshes.bin": ROOT / "data" / "neuromechfly" / "meshes.bin"}


SOURCES = ("kinematic", "physics", "policy")
POLICY: Path | None = None   # set by --policy


def make_replay(target_id: str, source: str = "kinematic") -> dict:
    target = get_target(target_id)
    if source == "kinematic":
        replay = scripted_episode(target_id, target["text"])
    elif source == "physics":
        from .physics import physics_episode
        replay = physics_episode(target_id, target["text"])
        replay["metadata"]["key_pitch_mm"] = KEY_PITCH_MM
    elif source == "policy":
        if POLICY is None:
            raise ValueError("start the server with --policy <checkpoint.zip> to replay a learned policy")
        from .policy_replay import policy_episode
        replay = policy_episode(POLICY, target_id, target["text"])
    else:
        raise ValueError(f"source must be one of {SOURCES}")
    replay["layout"] = layout_payload()
    if replay_text(replay["events"]) != replay["final_text"]:
        raise RuntimeError("Replay failed integrity check")
    replay["validation"] = judge_text(replay["final_text"], target_id)
    return replay


CACHE = ROOT / "data" / "replays" / "cache"
CACHE_INPUTS = [ROOT / "backend" / name for name in ("embodiment.py", "flybody.py", "keyboard.py", "judge.py")]
CACHE_INPUTS.append(FLY_ASSETS["/api/fly/model.json"])
SOURCE_INPUTS = {"kinematic": [], "physics": [ROOT / "backend" / "physics.py"],
                 "policy": [ROOT / "backend" / n for n in ("physics.py", "fly_env.py", "policy_replay.py")]}


@lru_cache(maxsize=8)
def replay_gzip(target_id: str, source: str = "kinematic") -> bytes:
    """Gzipped replay JSON, cached on disk by target text, source, and generator code."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    digest = hashlib.sha256(get_target(target_id)["text"].encode("utf-8"))
    for path in CACHE_INPUTS + SOURCE_INPUTS[source] + ([POLICY] if source == "policy" and POLICY else []):
        digest.update(path.read_bytes())
    stem = target_id if source == "kinematic" else f"{target_id}.{source}"
    cached = CACHE / f"{stem}-{digest.hexdigest()[:16]}.json.gz"
    if cached.is_file():
        return cached.read_bytes()
    payload = gzip.compress(json.dumps(make_replay(target_id, source), separators=(",", ":")).encode("utf-8"), 6)
    CACHE.mkdir(parents=True, exist_ok=True)
    for stale in CACHE.glob(f"{stem}-*.json.gz"):
        stale.unlink()
    cached.write_bytes(payload)
    return payload


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value: object, code: int = 200) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/targets":
            self.send_json(list_targets())
            return
        if parsed.path == "/api/sources":
            self.send_json([s for s in SOURCES if s != "policy" or POLICY is not None])
            return
        if parsed.path == "/api/replay":
            query = parse_qs(parsed.query)
            target_id = query.get("target_id", ["fly_demo"])[0]
            source = query.get("source", ["kinematic"])[0]
            try:
                payload = replay_gzip(target_id, source)
            except (KeyError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            if "gzip" in self.headers.get("Accept-Encoding", ""):
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_json(json.loads(gzip.decompress(payload)))
            return
        if parsed.path in FLY_ASSETS:
            self.send_file(FLY_ASSETS[parsed.path])
            return
        path = (WEB / parsed.path.lstrip("/")).resolve()
        if parsed.path == "/":
            path = WEB / "index.html"
        if not path.is_relative_to(WEB.resolve()) or not path.is_file():
            self.send_error(404)
            return
        self.send_file(path)

    def send_file(self, path: Path) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", type=Path, help="PPO/imitation checkpoint (.zip) for source=policy replays")
    args = parser.parse_args()
    global POLICY
    POLICY = args.policy.resolve() if args.policy else None
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FlySlop at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
