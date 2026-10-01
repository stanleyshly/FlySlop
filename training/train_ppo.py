"""Train a PPO policy on ``KeyboardTypingEnv`` with a token curriculum.

Example (full run, one seed)::

    uv run --extra training python -m training.train_ppo --seed 0

Smoke test (tiny budget, verifies the pipeline end to end)::

    uv run --extra training python -m training.train_ppo --smoke

Each run writes ``runs/ppo/<name>/`` with ``manifest.json`` (config, code,
dataset hashes, seed, versions), ``eval.jsonl`` (held-out progress),
``checkpoints/``, ``best_model.zip``, ``final_model.zip`` and TensorBoard logs.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from backend.memory_budget import GB, check_fits, start_watchdog, worker_budget
from training.common import (ROOT, DEFAULT_CONFIG, code_hash, config_hash, eval_targets, git_commit,
                             load_config, make_env, run_episode, summarize, versions)
from training.tokens import single_characters, splits_hash, token_splits


POLICY_TYPES = ("mlp", "connectome")


def policy_spec(config: dict) -> dict:
    """``config["policy"]``: ``{"type": "mlp"|"connectome", "connectome": {<connectome_kwargs>}}`` (default mlp)."""
    spec = dict(config.get("policy") or {})
    spec.setdefault("type", "mlp")
    if spec["type"] not in POLICY_TYPES:
        raise ValueError(f"policy type must be one of {POLICY_TYPES}, got {spec['type']!r}")
    return spec


def make_model(config: dict, vec, seed: int | None = None, device: str | None = None,
               tensorboard_log: str | None = None, verbose: int = 0) -> PPO:
    """PPO with the policy selected by ``config["policy"]`` (``mlp`` or ``connectome``).

    The connectome actor's device is resolved (``auto|cpu|mps``) and written back into the config's
    connectome kwargs, so the saved checkpoint records the concrete device/backend.
    """
    seed = config["seed"] if seed is None else seed
    ppo_kwargs = dict(config["ppo"])
    policy_kwargs = ppo_kwargs.pop("policy_kwargs", None) or {}
    spec = policy_spec(config)
    device = device or config.get("device", "cpu")
    if spec["type"] == "mlp":
        return PPO("MlpPolicy", vec, seed=seed, device=device, verbose=verbose, tensorboard_log=tensorboard_log,
                   policy_kwargs=policy_kwargs or None, **ppo_kwargs)
    from backend.connectome.device import resolve_device
    from training.connectome_policy import ConnectomeActorCriticPolicy, resolve_circuit
    kwargs = dict(spec.get("connectome") or {})
    kwargs.setdefault("seed", seed)
    arrays, _hash = resolve_circuit(kwargs.get("circuit", "data/connectome/typing_circuit.npz"))
    n = int(arrays["indptr"].shape[0] - 1)
    kwargs["device"] = resolve_device(kwargs.get("device", "auto"), n)
    # actor activations kept for backward: ~ minibatch * n * 4 B * (k_steps + few) * a few tensors
    est = int(ppo_kwargs.get("batch_size", 64)) * n * 4 * (int(kwargs.get("k_steps", 4)) + 4) * 3
    check_fits(est, f"connectome PPO minibatch ({ppo_kwargs.get('batch_size')} x {n} neurons)")
    vf = (policy_kwargs.get("net_arch") or {}).get("vf") if isinstance(policy_kwargs.get("net_arch"), dict) else None
    pk = {"connectome_kwargs": kwargs, "critic_arch": tuple(vf or (256, 256))}
    return PPO(ConnectomeActorCriticPolicy, vec, seed=seed, device=kwargs["device"], verbose=verbose,
               tensorboard_log=tensorboard_log, policy_kwargs=pk, **ppo_kwargs)


def load_ppo(path, device: str = "cpu") -> PPO:
    """``PPO.load`` that also works for connectome checkpoints. A checkpoint is rebuilt from its saved
    ``policy_kwargs`` (circuit path, variant, seed, readout, ...); the recorded connectome device/backend is
    overridden by ``device`` so an mps-trained model loads on cpu (the state_dict is backend independent)."""
    from stable_baselines3.common.save_util import load_from_zip_file
    data, _params, _vars = load_from_zip_file(path, device="cpu", load_data=True)
    custom = {}
    pk = (data or {}).get("policy_kwargs") or {}
    if "connectome_kwargs" in pk:
        ck = dict(pk["connectome_kwargs"])
        ck["device"] = device
        if ck.get("backend") == "index_add" and device == "cpu":
            ck["backend"] = "auto"
        custom["policy_kwargs"] = {**pk, "connectome_kwargs": ck}
    return PPO.load(path, device=device, custom_objects=custom or None)


def curriculum_pool(stage: dict, splits: dict) -> list[str]:
    if stage["pool"] == "chars":
        return single_characters(splits["train"])
    pool = [t for t in splits[stage["pool"]] if len(t) <= stage.get("max_len", 10**9)]
    if not pool:
        raise ValueError(f"empty curriculum pool: {stage}")
    return pool


class CurriculumAndEval(BaseCallback):
    """Swap target pools at curriculum boundaries; evaluate held-out tokens periodically."""

    def __init__(self, config: dict, splits: dict, run_dir: Path, total: int):
        super().__init__()
        self.config, self.splits, self.run_dir, self.total = config, splits, run_dir, total
        self.stage_index = -1
        self.next_eval = 0
        self.last_eval = -1
        self.best = -1.0
        self.eval_env = make_env(config, splits["heldout"])
        self.eval_log = (run_dir / "eval.jsonl").open("a", encoding="utf-8")
        self.started = time.time()

    def _set_stage(self) -> None:
        fraction = self.num_timesteps / self.total
        stages = self.config["curriculum"]
        index = next((i for i, s in enumerate(stages) if fraction < s["until_fraction"]), len(stages) - 1)
        if index != self.stage_index:
            self.stage_index = index
            pool = curriculum_pool(stages[index], self.splits)
            self.training_env.env_method("set_targets", pool)
            self.logger.record("curriculum/stage", index)
            print(f"[curriculum] step={self.num_timesteps} stage={index} pool={len(pool)} targets", flush=True)

    def _evaluate(self) -> None:
        cfg = self.config["eval"]
        plan = eval_targets(self.splits["heldout"], cfg["episodes"], seed=10_000 + self.config["seed"])
        policy = lambda obs, _env: self.model.predict(obs, deterministic=True)[0]
        summary = summarize([run_episode(self.eval_env, t, s, policy) for t, s in plan])
        self.last_eval = self.num_timesteps
        record = {"timesteps": self.num_timesteps, "stage": self.stage_index,
                  "wall_seconds": round(time.time() - self.started, 1), **summary}
        self.eval_log.write(json.dumps(record) + "\n")
        self.eval_log.flush()
        for key in ("exact_match", "mean_cer", "mean_ticks"):
            self.logger.record(f"heldout/{key}", summary[key])
        print(f"[eval] step={self.num_timesteps} heldout exact={summary['exact_match']:.3f} "
              f"cer={summary['mean_cer']:.3f} ticks={summary['mean_ticks']:.1f}", flush=True)
        if summary["exact_match"] > self.best:
            self.best = summary["exact_match"]
            self.model.save(self.run_dir / "best_model.zip")

    def _on_training_start(self) -> None:
        self._set_stage()

    def _on_step(self) -> bool:
        self._set_stage()
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.config["eval"]["freq_timesteps"]
        return True

    def _on_training_end(self) -> None:
        if self.last_eval != self.num_timesteps:
            self._evaluate()
        self.eval_log.close()


def apply_smoke(config: dict) -> dict:
    config = copy.deepcopy(config)
    config.update(total_timesteps=8192, n_envs=4, checkpoint_freq_timesteps=4096)
    config["ppo"].update(n_steps=256, batch_size=256, n_epochs=2)
    config["eval"] = {"freq_timesteps": 4096, "episodes": 4 if "fly_env" in config["environment"] else 10}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--device", help="PPO device for the mlp policy; connectome uses --connectome-device")
    parser.add_argument("--policy", choices=POLICY_TYPES, help="actor: mlp or connectome (overrides config)")
    parser.add_argument("--variant", choices=("real", "shuffled", "random_sparse", "frozen"),
                        help="connectome wiring variant")
    parser.add_argument("--circuit", help="connectome circuit (.npz typing circuit or legacy .json)")
    parser.add_argument("--connectome-device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (default from config)")
    parser.add_argument("--init-from", help="PPO/BC checkpoint whose policy weights initialise training")
    parser.add_argument("--smoke", action="store_true", help="tiny budget pipeline check")
    parser.add_argument("--out", default=str(ROOT / "runs" / "ppo"))
    parser.add_argument("--name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = apply_smoke(config)
    for key, value in (("seed", args.seed), ("total_timesteps", args.timesteps),
                       ("n_envs", args.n_envs), ("device", args.device)):
        if value is not None:
            config[key] = value
    if args.policy or args.variant or args.circuit or args.connectome_device:
        spec = config.setdefault("policy", {"type": "mlp"})
        conn = spec.setdefault("connectome", {})
        if args.policy:
            spec["type"] = args.policy
        elif args.variant or args.circuit or args.connectome_device:
            spec["type"] = "connectome"
        for key, value in (("variant", args.variant), ("circuit", args.circuit),
                           ("device", args.connectome_device)):
            if value:
                conn[key] = value
    policy_spec(config)
    start_watchdog(config_gb=None)
    fit_workers = worker_budget(int(0.5 * GB), reserve_bytes=int(0.8 * GB))
    if config["n_envs"] > max(1, fit_workers):
        print(f"[ram] n_envs {config['n_envs']} -> {max(1, fit_workers)} to fit the RAM cap", flush=True)
        config["n_envs"] = max(1, fit_workers)
    seed, total = config["seed"], config["total_timesteps"]
    name = args.name or f"{'smoke' if args.smoke else 'ppo'}-seed{seed}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=False)

    splits = token_splits(**config["tokens"])
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    manifest = {
        "run": name, "created": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "config": config, "config_sha256": config_hash(config), "code_sha256": code_hash(),
        "git_commit": git_commit(), "dataset_sha256": splits_hash(splits),
        "split_sizes": {k: len(v) for k, v in splits.items()}, "versions": versions(),
        "smoke": args.smoke, "learning_claim": False,
        "claim_note": config["claim"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    subproc = args.subproc or config.get("subproc", False)
    torch.set_num_threads(1 if subproc else torch.get_num_threads())
    first_pool = curriculum_pool(config["curriculum"][0], splits)
    factories = [lambda i=i: Monitor(make_env(config, first_pool)) for i in range(config["n_envs"])]
    vec = (SubprocVecEnv if subproc else DummyVecEnv)(factories)
    vec.seed(seed)

    model = make_model(config, vec, seed=seed, device=config["device"], tensorboard_log=str(run_dir / "tb"))
    if policy_spec(config)["type"] == "connectome":
        actor = model.policy.actor
        manifest["connectome"] = {"kwargs": model.policy.connectome_kwargs, "circuit_content_hash": actor.circuit_hash,
                                  "backend": actor.runtime.backend, "n_neurons": actor.runtime.n,
                                  "signature": actor.runtime.variant_signature()}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.init_from:
        init = load_ppo(args.init_from, device=config["device"])
        model.policy.load_state_dict(init.policy.state_dict())
        manifest["init_from"] = str(args.init_from)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    callbacks = CallbackList([
        CurriculumAndEval(config, splits, run_dir, total),
        CheckpointCallback(max(1, config["checkpoint_freq_timesteps"] // config["n_envs"]),
                           str(run_dir / "checkpoints"), name_prefix="ppo"),
    ])
    started = time.time()
    model.learn(total_timesteps=total, callback=callbacks, progress_bar=False)
    model.save(run_dir / "final_model.zip")
    vec.close()

    manifest.update(finished=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=round(time.time() - started, 1),
                    steps_per_second=round(total / max(1e-9, time.time() - started), 1))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "wall_seconds": manifest["wall_seconds"],
                      "steps_per_second": manifest["steps_per_second"]}, indent=2))


if __name__ == "__main__":
    main()
