"""Generate a portable replay JSON and verify its contact-to-text chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embodiment import replay_text
from .server import make_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="fly_demo")
    parser.add_argument("--output", type=Path, default=Path("data/replays/fly_demo.json"))
    parser.add_argument("--source", choices=("kinematic", "physics"), default="kinematic")
    args = parser.parse_args()
    payload = make_replay(args.target, args.source)
    assert replay_text(payload["events"]) == payload["final_text"] == payload["target"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(payload['events'])} events, {len(payload['final_text'])} characters")
    print(json.dumps(payload["validation"], indent=2))


if __name__ == "__main__":
    main()
