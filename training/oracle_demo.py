"""Write a deterministic scripted contact demonstration (not a policy run)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.embodiment import scripted_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default="fly")
    parser.add_argument("--target-id", default="authored_oracle_demo")
    parser.add_argument("--output", type=Path, default=Path("training/oracle_demo.json"))
    args = parser.parse_args()
    episode = scripted_episode(args.target_id, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": episode["metadata"]["mode"], "target": args.target,
                      "final_text": episode["final_text"], "events": len(episode["events"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
