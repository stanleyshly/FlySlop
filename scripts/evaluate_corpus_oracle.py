"""Reproduce the deterministic contact-path oracle result for every target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.corpus import get_target, list_targets
from backend.embodiment import replay_text, scripted_episode
from backend.judge import judge_text


def main() -> None:
    results = []
    for item in list_targets():
        target = get_target(item["id"])
        replay = scripted_episode(item["id"], target["text"])
        reconstructed = replay_text(replay["events"])
        judged = judge_text(reconstructed, item["id"])
        result = {
            "target_id": item["id"],
            "characters": len(target["text"]),
            "keypresses": sum(event["type"] == "key" for event in replay["events"]),
            "exact": judged["exact"]["passed"],
            "replay_matches": reconstructed == replay["final_text"],
            "syntax": judged["syntax"]["passed"],
            "elaboration": judged["elaboration"]["passed"],
        }
        results.append(result)
    summary = {
        "mode": "scripted_kinematic_oracle",
        "learned": False,
        "targets": len(results),
        "exact": sum(result["exact"] for result in results),
        "replay_matches": sum(result["replay_matches"] for result in results),
        "syntax": sum(result["syntax"] for result in results),
        "elaboration": sum(result["elaboration"] for result in results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if not all(result["exact"] and result["replay_matches"] and result["syntax"] and result["elaboration"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
