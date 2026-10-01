"""Deterministic contact-path check; this is an oracle baseline, not learned eval."""

from __future__ import annotations

import argparse
import json

from backend.keyboard import BY_ID, CHAR_TO_KEY
from backend.env import KeyboardReachEnv


def episode(target: str, seed: int = 0) -> dict:
    env = KeyboardReachEnv(targets=(target,), max_steps=max(10, len(target) * 12))
    env.reset(seed=seed, options={"target": target})
    total_reward = 0.0
    done = False
    # Scripted inverse lookup chooses a key from the known target and drives
    # travel, press, hold, release through the environment action interface.
    for char in target:
        key_id, needs_shift = CHAR_TO_KEY[char]
        for physical_key_id in (["ShiftLeft", key_id] if needs_shift else [key_id]):
            key = BY_ID[physical_key_id]
            x, y = key.x + key.width / 2, key.y + key.height / 2
            for z in (0.6, 0.15, -0.1, -0.1, 0.6):
                _obs, reward, terminated, truncated, _info = env.step([x, y, z])
                total_reward += reward
                done = terminated or truncated
                if done:
                    break
            if done:
                break
        if done:
            break
    exact = env.buffer == target
    result = {"seed": seed, "target": target, "text": env.buffer, "exact": exact,
              "ticks": env.tick, "reward": total_reward,
              "contact_events": sum(e["type"].startswith("contact_") for e in env.events),
              "mode": "deterministic_oracle_baseline", "learned": False}
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=["fly", "abc", "sv"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = [episode(target, args.seed) for target in args.targets]
    print(json.dumps({"evaluation": "oracle contact-path smoke evaluation",
                      "learning_claim": False, "results": results}, indent=2))


if __name__ == "__main__":
    main()
