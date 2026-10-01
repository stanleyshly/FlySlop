"""Paired connectome ablation (PLAN §5.1 step 5, package E).

Every variant gets the same demonstrations, the same imitation budget, the
same seeds, and the same held-out evaluation episodes; only the actor wiring
differs. Variants:

- ``mlp``: ordinary 256x256 MLP actor (reference, larger capacity)
- ``connectome``: MaleCNS right-foreleg circuit, measured sparsity and signs
- ``shuffled``: same blocks, edge count, weights and signs, positions permuted
- ``random_sparse``: same density, random positions across allowed blocks
- ``dense``: same neurons, all allowed blocks connected, no sign prior
- ``silenced``: connectome actor with proprioceptive/tactile inputs zeroed
- ``random``: uniform random actions (no training)

The predeclared metric is held-out exact match; validation action MSE on
held-out demonstrations is secondary. The low-data default (60 demo episodes)
avoids a ceiling where every variant imitates perfectly. Reported: per-seed
rows, median and range per variant, and a bootstrap 95% interval for the
paired difference connectome minus each control.

    uv run --extra training python -m training.connectome_experiment --seeds 0 1 2 3 4 --out runs/connectome/exp1
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from backend.connectome import load_circuit, summary
from training.common import (code_hash, eval_targets, load_config, make_env, make_model, run_episode, summarize,
                             versions)
from training.imitate import collect, fit
from training.tokens import single_characters, splits_hash, token_splits

VARIANTS = ("mlp", "connectome", "shuffled", "random_sparse", "dense", "silenced", "random")


def evaluate(config, splits, act, episodes: int, seed: int) -> dict:
    env = make_env(config, splits["heldout"])
    plan = eval_targets(splits["heldout"], episodes, seed=20_000 + seed)
    return summarize([run_episode(env, t, s, act) for t, s in plan])


def run_variant(job) -> dict:
    config, variant, seed, data, splits, episodes, pool = job
    torch.set_num_threads(1)
    started = time.time()
    if variant == "random":
        rng = np.random.default_rng(seed)
        result = evaluate(config, splits, lambda _o, _e: rng.uniform(-1, 1, 8).astype(np.float32), episodes, seed)
        return {"variant": variant, "seed": seed, "heldout": result, "val_mse": None, "params": 0,
                "seconds": round(time.time() - started, 1)}
    obs, act, val_obs, val_act = data
    model = make_model(config, variant, seed, pool, device="cpu", verbose=0)
    losses = fit(model, obs, act, config["imitation"], seed)
    with torch.no_grad():
        mean = model.policy.get_distribution(torch.as_tensor(val_obs)).distribution.mean
        val_mse = float(torch.nn.functional.mse_loss(mean, torch.as_tensor(val_act)))
    actor_params = sum(p.numel() for n, p in model.policy.named_parameters()
                       if not n.startswith(("value_net", "mlp_extractor.value", "mlp_extractor.critic")))
    result = evaluate(config, splits, lambda o, _e: model.predict(o, deterministic=True)[0], episodes, seed)
    return {"variant": variant, "seed": seed, "heldout": result, "val_mse": val_mse, "train_mse": losses[-1],
            "params": int(actor_params), "seconds": round(time.time() - started, 1)}


def bootstrap_diff(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> list[float]:
    diff = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="training/ppo_fly.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--demo-episodes", type=int, default=60)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-episodes", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["imitation"]["epochs"] = args.epochs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    splits = token_splits(**config["tokens"])
    pool = splits["train"] + single_characters(splits["train"])
    started = time.time()
    obs, act, stats = collect(config, pool, args.demo_episodes, 777, args.workers)
    val_obs, val_act, _ = collect(dict(config, imitation={**config["imitation"], "noisy_fraction": 0.0}),
                                  splits["heldout"], args.val_episodes, 778, args.workers)
    print(f"[demos] train {len(obs)} transitions, val {len(val_obs)} ({time.time() - started:.0f}s)", flush=True)
    data = (obs, act, val_obs, val_act)
    jobs = [(config, v, s, data, splits, args.eval_episodes, pool) for s in args.seeds for v in args.variants]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for row in ex.map(run_variant, jobs):
            rows.append(row)
            print(f"[{row['variant']:>13} seed {row['seed']}] heldout exact {row['heldout']['exact_match']:.3f} "
                  f"cer {row['heldout']['mean_cer']:.3f} val_mse {row['val_mse']} ({row['seconds']}s)", flush=True)

    table = {}
    for v in args.variants:
        scores = [r["heldout"]["exact_match"] for r in rows if r["variant"] == v]
        mses = [r["val_mse"] for r in rows if r["variant"] == v and r["val_mse"] is not None]
        table[v] = {"median_exact": float(np.median(scores)), "min_exact": min(scores), "max_exact": max(scores),
                    "median_val_mse": float(np.median(mses)) if mses else None,
                    "actor_params": next(r["params"] for r in rows if r["variant"] == v)}
    comparisons = {}
    if "connectome" in args.variants:
        base = {r["seed"]: r["heldout"]["exact_match"] for r in rows if r["variant"] == "connectome"}
        for v in args.variants:
            if v == "connectome":
                continue
            other = {r["seed"]: r["heldout"]["exact_match"] for r in rows if r["variant"] == v}
            seeds = sorted(set(base) & set(other))
            a, b = [base[s] for s in seeds], [other[s] for s in seeds]
            comparisons[f"connectome_minus_{v}"] = {"mean": float(np.mean(np.array(a) - np.array(b))),
                                                    "ci95": bootstrap_diff(a, b), "n_seeds": len(seeds)}
    circuit = load_circuit()
    report = {"experiment": "connectome ablation, imitation training, state-assisted physics body",
              "metric": "held-out exact match (predeclared); validation action MSE secondary",
              "circuit": {"file_sha256": circuit["file_sha256"], **summary(circuit), "source": circuit["source"]},
              "demonstrations": stats | {"val_transitions": int(len(val_obs))},
              "config": config, "code_sha256": code_hash(), "dataset_sha256": splits_hash(splits),
              "versions": versions(), "seeds": args.seeds, "per_run": rows, "table": table,
              "paired_differences": comparisons, "wall_seconds": round(time.time() - started, 1),
              "claim_rule": "A connectome benefit is claimed only if every paired 95% interval against "
                            "shuffled, random_sparse and dense excludes zero in favour of the connectome."}
    claim = all(comparisons.get(f"connectome_minus_{v}", {}).get("ci95", [0])[0] > 0
                for v in ("shuffled", "random_sparse", "dense") if v in args.variants)
    report["connectome_benefit_supported"] = bool(comparisons) and claim
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"table": table, "paired_differences": comparisons,
                      "connectome_benefit_supported": report["connectome_benefit_supported"]}, indent=2))


if __name__ == "__main__":
    main()
