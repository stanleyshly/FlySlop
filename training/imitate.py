"""Supervised imitation of the scripted controller (PLAN §5.1 step 3).

Collects demonstrations from the environment's scripted expert on training
targets, then fits the actor of an SB3 PPO policy to the expert actions. Part
of the episodes are run with noisy executed actions while still labelling the
clean expert action, so the policy sees (and learns to recover from) states
off the expert's path. The result is saved as a normal PPO checkpoint that
``training.train_ppo --init-from`` can fine-tune.

    uv run --extra training python -m training.imitate --config training/ppo_fly.json --out runs/bc/fly-seed0
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from backend.memory_budget import GB, start_watchdog, worker_budget
from training.common import (DEFAULT_CONFIG, code_hash, config_hash, eval_targets, load_config, make_env,
                             run_episode, summarize, versions)
from training.common import make_model as make_legacy_model
from training.train_ppo import make_model
from training.tokens import single_characters, splits_hash, token_splits


def demo_worker(args) -> tuple[np.ndarray, np.ndarray, int, int]:
    config, targets, seed, noise = args
    rng = np.random.default_rng(seed)
    env = make_env(config, targets)
    obs_list, act_list, solved = [], [], 0
    for i, target in enumerate(targets):
        obs, _ = env.reset(seed=seed * 10_000 + i, options={"target": target})
        done = False
        while not done:
            expert = env.expert_action()
            obs_list.append(obs)
            act_list.append(expert)
            executed = np.clip(expert + rng.normal(0, noise, expert.shape), -1, 1) if noise else expert
            obs, _r, terminated, truncated, _info = env.step(executed)
            done = terminated or truncated
        solved += int(env.typed == target)
    return np.array(obs_list, np.float32), np.array(act_list, np.float32), solved, len(targets)


def collect(config: dict, pool: list[str], episodes: int, seed: int, workers: int,
            progress=None) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = config["imitation"]
    rng = np.random.default_rng(seed)
    targets = [pool[int(rng.integers(len(pool)))] for _ in range(episodes)]
    chunks = np.array_split(np.arange(episodes), workers)
    noisy_cut = int(len(chunks) * cfg["noisy_fraction"])
    jobs = [(config, [targets[i] for i in chunk], seed * 100 + k, cfg["action_noise"] if k < noisy_cut else 0.0)
            for k, chunk in enumerate(chunks) if len(chunk)]
    parts = [None] * len(jobs)
    with ProcessPoolExecutor(len(jobs)) as ex:
        futures = {ex.submit(demo_worker, job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            parts[i] = future.result()
            if progress:
                progress(done, len(jobs), int(len(parts[i][0])), len(jobs[i][1]))
    obs = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    stats = {"episodes": episodes, "transitions": int(len(obs)),
             "expert_exact_while_collecting": sum(p[2] for p in parts) / max(1, sum(p[3] for p in parts))}
    return obs, act, stats


def fit(model, obs: np.ndarray, act: np.ndarray, cfg: dict, seed: int, details: list | None = None) -> list[float]:
    """Behaviour-cloning MSE on the policy's action mean. With ``action_mode="mn"`` the 16-D action is the thorax
    velocity plus the 14 expert joint angles (``LegIK`` of the expert's claw targets, normalised by ``JOINT_SCALE``),
    so this is an MSE on joint-space targets. ``details`` (if given) receives per-epoch ``{total, velocity, joints}``
    MSEs (``joints`` is None for 8-D claw actions). ``cfg["max_seconds"]`` stops early."""
    policy = model.policy
    started = time.time()
    torch.manual_seed(seed)
    critic = ("value_net", "mlp_extractor.value", "mlp_extractor.critic")
    params = [p for name, p in policy.named_parameters() if not name.startswith(critic) and name != "log_std"]
    optim = torch.optim.Adam(params, lr=cfg["learning_rate"])
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(act)
    losses = []
    n = len(obs_t)
    for _ in range(cfg["epochs"]):
        order = torch.randperm(n)
        total, parts = 0.0, np.zeros(2)
        for start in range(0, n, cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            mean = policy.get_distribution(obs_t[idx]).distribution.mean
            loss = torch.nn.functional.mse_loss(mean, act_t[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach()) * len(idx)
            if act_t.shape[1] == 16:
                sq = ((mean.detach() - act_t[idx]) ** 2).sum(0)
                parts += [float(sq[:2].sum()), float(sq[2:].sum())]
        losses.append(total / n)
        if details is not None:
            mn = act_t.shape[1] == 16
            details.append({"total": losses[-1], "velocity": parts[0] / (2 * n) if mn else None,
                            "joints": parts[1] / (14 * n) if mn else None, "seconds": round(time.time() - started, 1)})
        if cfg.get("max_seconds") and time.time() - started > cfg["max_seconds"]:
            break
    with torch.no_grad():
        policy.log_std.fill_(cfg["log_std_after"])
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy", default="mlp",
                        help="mlp | connectome (SparseRecurrent runtime, see --variant); legacy 360-neuron actors: "
                             "shuffled | random_sparse | dense | silenced (training.common.make_model)")
    parser.add_argument("--variant", choices=("real", "shuffled", "random_sparse", "frozen"),
                        help="wiring variant of --policy connectome")
    parser.add_argument("--circuit", help="connectome circuit (.npz or legacy .json)")
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda", "mps"),
                        help="device of the connectome actor (mlp always runs on cpu)")
    parser.add_argument("--action-mode", choices=("claw", "mn"),
                        help="env action space: claw (8-D claw targets) or mn (thorax velocity + 14 joint targets)")
    parser.add_argument("--readout", help="connectome readout name (e.g. mn_antagonist; default mn_dn)")
    parser.add_argument("--chars", help="train and evaluate on single characters from this string (e.g. 'asdfjkl;')")
    parser.add_argument("--no-state-assist", action="store_true", help="hide the next-key offsets from the observation")
    parser.add_argument("--editor-obs", action="store_true", help="add editor cursor/buffer-tail observations")
    parser.add_argument("--max-fit-seconds", type=float, help="stop the imitation fit after this many seconds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.episodes is not None:
        config["imitation"]["episodes"] = args.episodes
    if args.epochs is not None:
        config["imitation"]["epochs"] = args.epochs
    if args.max_fit_seconds:
        config["imitation"]["max_seconds"] = args.max_fit_seconds
    for flag, key, value in (("action_mode", "action_mode", args.action_mode), ("no_state_assist", "state_assist", False),
                             ("editor_obs", "editor_obs", True)):
        if getattr(args, flag):
            config["env"][key] = value
    seed = config["seed"]
    start_watchdog()
    workers = max(1, min(args.workers, worker_budget(int(0.5 * GB), reserve_bytes=int(0.8 * GB))))
    if workers != args.workers:
        print(f"[ram] workers {args.workers} -> {workers} to fit the RAM cap", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.chars:
        splits = {"train": sorted(set(args.chars)), "heldout": sorted(set(args.chars))}
        pool = splits["train"]
    else:
        splits = token_splits(**config["tokens"])
        pool = splits["train"] + single_characters(splits["train"])
    started = time.time()
    obs, act, stats = collect(config, pool, config["imitation"]["episodes"], seed, workers)
    print(f"[imitate] {stats['transitions']} transitions from {stats['episodes']} episodes "
          f"(expert exact {stats['expert_exact_while_collecting']:.2f}) in {time.time() - started:.0f}s", flush=True)

    if args.policy in ("mlp", "connectome"):
        if args.policy == "connectome":
            conn = config.setdefault("policy", {}).setdefault("connectome", {})
            config["policy"]["type"] = "connectome"
            for key, value in (("variant", args.variant), ("circuit", args.circuit), ("device", args.device),
                               ("readout", args.readout)):
                if value:
                    conn[key] = value
        else:
            config["policy"] = {"type": "mlp"}
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec = DummyVecEnv([lambda: make_env(config, pool)])
        model = make_model(config, vec, seed=seed, device="cpu")
    else:
        model = make_legacy_model(config, args.policy, seed, pool, device="cpu", verbose=0)
    details: list = []
    fit_started = time.time()
    losses = fit(model, obs, act, config["imitation"], seed, details)
    print(f"[imitate] action MSE {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} epochs "
          f"in {time.time() - fit_started:.0f}s", flush=True)
    if details[-1]["joints"] is not None:
        print(f"[imitate] joint-angle MSE {details[0]['joints']:.4f} -> {details[-1]['joints']:.4f}; "
              f"velocity MSE {details[0]['velocity']:.4f} -> {details[-1]['velocity']:.4f}", flush=True)
    model.save(out / "bc_model.zip")
    model.get_env().close()
    connectome = None
    if hasattr(model.policy, "connectome_kwargs"):
        actor = model.policy.actor
        connectome = {"kwargs": model.policy.connectome_kwargs, "circuit_content_hash": actor.circuit_hash,
                      "backend": actor.runtime.backend, "signature": actor.runtime.variant_signature()}

    env = make_env(config, splits["heldout"])
    policy = lambda o, _e: model.predict(o, deterministic=True)[0]
    plan = eval_targets(splits["heldout"], args.eval_episodes, seed=10_000 + seed)
    episodes = [run_episode(env, t, s, policy) for t, s in plan]
    heldout = summarize(episodes)
    heldout["chars_target"] = sum(len(e["target"]) for e in episodes)
    heldout["chars_typed"] = sum(len(e["text"]) for e in episodes)
    heldout["chars_correct"] = sum(next((i for i, (a, b) in enumerate(zip(e["target"], e["text"])) if a != b),
                                        min(len(e["target"]), len(e["text"]))) for e in episodes)
    print(f"[imitate] held-out exact {heldout['exact_match']:.3f} cer {heldout['mean_cer']:.3f} "
          f"chars typed {heldout['chars_typed']} correct {heldout['chars_correct']} of {heldout['chars_target']}",
          flush=True)
    (out / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    manifest = {"run": out.name, "kind": "behaviour_cloning", "policy": args.policy, "seed": seed, "config": config,
                "config_sha256": config_hash(config), "code_sha256": code_hash(), "dataset_sha256": splits_hash(splits),
                "connectome": connectome, "demonstrations": stats, "loss_curve": losses, "loss_details": details, "heldout": heldout, "versions": versions(),
                "wall_seconds": round(time.time() - started, 1), "learning_claim": False,
                "claim_note": "Imitation of a scripted controller; see config claim."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "heldout_exact": heldout["exact_match"]}), flush=True)


if __name__ == "__main__":
    main()
