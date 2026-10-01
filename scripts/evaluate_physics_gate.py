"""PLAN §5.3 scripted-body gate on the MuJoCo keyboard.

Types seeded random character strings (drawn from the corpus alphabet) with
the scripted gait/press planner through physics and counts, per seed:
intended presses that produced a contact onset on the intended key,
characters that were not intended, and uncredited key actuations (a key
pushed past the threshold without a claw touching it).

    uv run python scripts/evaluate_physics_gate.py --presses 200 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.corpus import get_target, list_targets  # noqa: E402
from backend.embodiment import key_sequence, replay_text  # noqa: E402
from backend.physics import physics_episode, PhysicsParams  # noqa: E402
from dataclasses import asdict  # noqa: E402


def alphabet() -> list[str]:
    chars = set()
    for item in list_targets():
        chars |= set(get_target(item["id"])["text"])
    return sorted(c for c in chars if c not in "\n")


def random_text(seed: int, presses: int) -> str:
    rng = np.random.default_rng(seed)
    chars, text = alphabet(), ""
    while len(key_sequence(text)) < presses:
        text += chars[int(rng.integers(len(chars)))]
    return text


def score(seed: int, presses: int) -> dict:
    text = random_text(seed, presses)
    intended = key_sequence(text)
    result = physics_episode(f"gate-seed{seed}", text)
    onsets = [e["key_id"] for e in result["events"] if e["type"] == "contact_onset"]
    # Align onsets to intended presses in order (greedy subsequence match).
    matched, j = 0, 0
    for key in intended:
        while j < len(onsets) and onsets[j] != key:
            j += 1
        if j < len(onsets):
            matched += 1
            j += 1
    typed = result["final_text"]
    replay_ok = replay_text(result["events"]) == typed
    wrong = sum(1 for a, b in zip(typed, text) if a != b) + abs(len(typed) - len(text))
    return {"seed": seed, "intended_presses": len(intended), "contact_onsets": len(onsets),
            "intended_contacts": matched, "intended_contact_rate": matched / len(intended),
            "unintended_characters": max(0, len(onsets) - matched),
            "unintended_char_rate": max(0, len(onsets) - matched) / max(1, len(onsets)),
            "uncredited_actuations": result["metadata"]["unattributed_presses"],
            "exact_text": typed == text, "char_mismatches": wrong, "replay_ok": replay_ok,
            "ticks": len(result["frames"]["rows"]), "max_ik_error": result["metadata"]["max_ik_error"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presses", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output")
    args = parser.parse_args()
    with ProcessPoolExecutor(len(args.seeds)) as pool:
        rows = list(pool.map(score, args.seeds, [args.presses] * len(args.seeds)))
    intended = sum(r["intended_presses"] for r in rows)
    contacts = sum(r["intended_contacts"] for r in rows)
    onsets = sum(r["contact_onsets"] for r in rows)
    unintended = sum(r["unintended_characters"] for r in rows)
    summary = {"intended_contact_rate": contacts / intended,
               "unintended_char_rate": unintended / max(1, onsets),
               "uncredited_actuations": sum(r["uncredited_actuations"] for r in rows),
               "all_replays_ok": all(r["replay_ok"] for r in rows)}
    summary["gate_met"] = (summary["intended_contact_rate"] >= 0.95 and summary["unintended_char_rate"] <= 0.01
                           and summary["all_replays_ok"] and len(rows) >= 3)
    report = {"gate": "PLAN §5.3 scripted body: >=95% intended contacts, <=1% unintended characters, "
                      "200 presses x 3 seeds", "physics": asdict(PhysicsParams()), "per_seed": rows,
              "summary": summary}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
