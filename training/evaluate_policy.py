"""Evaluate trained PPO runs on held-out tokens against random and scripted baselines.

    uv run --extra training python -m training.evaluate_policy runs/ppo/ppo-seed0-*
    uv run --extra training python -m training.evaluate_policy runs/ppo/* --episodes 100 \
        --output runs/ppo/report.json

Every episode is replay-checked (``backend.embodiment.replay_text``). The
report lists per-run results and, across runs, median and range of held-out
exact match, as ``PLAN.md`` section 5 requires. The random baseline samples
uniform actions; the scripted baseline is a proportional reach/tap controller
driven through the same action interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from training.common import code_hash, eval_targets, make_env, run_episode, scripted_action, summarize


def evaluate(act, config: dict, tokens: list[str], episodes: int, seed: int, keep: bool = False) -> dict:
    env = make_env(config, tokens)
    rows = [run_episode(env, t, s, act) for t, s in eval_targets(tokens, episodes, seed)]
    out = summarize(rows)
    if keep:
        out["per_episode"] = rows
    return out


def replay_file(model, config: dict, target: str, seed: int, path: Path) -> None:
    """One replay-checked episode: viewer-format frames for the fly body, poses for the point foot."""
    fly = "fly_env" in config["environment"]
    env = make_env(config, [target], **({"record": True} if fly else {}))
    result = run_episode(env, target, seed, lambda obs, _e: model.predict(obs, deterministic=True)[0])
    if fly:
        payload = env.replay(target, "learned_policy_physics_body") | {"result": result}
    else:
        payload = {"metadata": {"mode": "learned_ppo_point_foot", "target": target, "seed": seed},
                   "result": result, "poses": env.poses, "events": env.events}
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories written by training.train_ppo")
    parser.add_argument("--model", default="best_model.zip", help="file inside each run dir")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--per-episode", action="store_true")
    parser.add_argument("--replay", help="write one held-out replay per run with this target")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=1, help="(reserved) evaluation is sequential per run")
    args = parser.parse_args()

    report = {"evaluation": "held-out token transcription, state-assisted",
              "code_sha256": code_hash(), "episodes_per_condition": args.episodes, "runs": []}
    baselines = None
    for run in map(Path, args.runs):
        manifest = json.loads((run / "manifest.json").read_text())
        config = manifest["config"]
        splits = json.loads((run / "splits.json").read_text())
        model = PPO.load(run / args.model, device="cpu")
        policy = lambda obs, _env: model.predict(obs, deterministic=True)[0]
        entry = {"run": str(run), "seed": manifest["seed"], "smoke": manifest.get("smoke", False),
                 "environment": config["environment"], "init_from": manifest.get("init_from"),
                 "dataset_sha256": manifest["dataset_sha256"], "code_sha256_at_train": manifest["code_sha256"],
                 "heldout": evaluate(policy, config, splits["heldout"], args.episodes, args.eval_seed, args.per_episode),
                 "train": evaluate(policy, config, splits["train"], args.episodes, args.eval_seed)}
        if baselines is None:
            rng = np.random.default_rng(args.eval_seed)
            random_act = lambda _obs, env: rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            baselines = {"random": evaluate(random_act, config, splits["heldout"], args.episodes, args.eval_seed),
                         "scripted": evaluate(lambda _o, env: scripted_action(env), config, splits["heldout"],
                                              args.episodes, args.eval_seed)}
        if args.replay:
            replay_file(model, config, args.replay, args.eval_seed, run / f"replay-{''.join(c if c.isalnum() else '_' for c in args.replay)}.json")
        report["runs"].append(entry)
        print(f"{run.name}: heldout exact={entry['heldout']['exact_match']:.3f} "
              f"cer={entry['heldout']['mean_cer']:.3f} train exact={entry['train']['exact_match']:.3f}", flush=True)

    scores = [r["heldout"]["exact_match"] for r in report["runs"]]
    report["baselines_heldout"] = baselines
    report["aggregate_heldout_exact"] = {"median": float(np.median(scores)), "min": min(scores),
                                         "max": max(scores), "n_runs": len(scores)}
    gate = 0.9
    report["gate"] = {"threshold": gate, "requires_seeds": 5,
                      "met": len(scores) >= 5 and min(scores) >= gate and not any(r["smoke"] for r in report["runs"])}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("baselines_heldout", "aggregate_heldout_exact", "gate")}, indent=2))


if __name__ == "__main__":
    main()
