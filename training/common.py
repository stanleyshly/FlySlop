"""Shared helpers for PPO training and evaluation runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from backend.embodiment import replay_text
from backend.env import key_center, physical_key_for

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "training" / "ppo.json"
CODE_FILES = ("backend/env.py", "backend/fly_env.py", "backend/physics.py", "backend/keyboard.py",
              "backend/embodiment.py", "backend/flybody.py", "training/common.py", "training/tokens.py",
              "training/train_ppo.py", "training/imitate.py", "training/evaluate_policy.py",
              "training/connectome_policy.py", "backend/editor.py", "backend/policy_replay.py")
CODE_GLOBS = ("backend/connectome/*.py",)
CIRCUIT_FILES = ("data/connectome/typing_circuit.npz", "data/connectome/rf_leg_circuit.json",
                 "data/connectome/mn_joint_map.json")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def code_hash() -> str:
    digest = hashlib.sha256()
    names = list(CODE_FILES)
    for pattern in CODE_GLOBS:
        names += sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern))
    for name in names:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    for name in CIRCUIT_FILES:            # circuit content, hashed by file bytes (missing file = empty)
        path = ROOT / name
        digest.update(name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest() if path.is_file() else b"")
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict:
    import gymnasium
    import stable_baselines3
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
            "platform": platform.platform()}


ENVIRONMENTS = {"backend.env.KeyboardTypingEnv": ("backend.env", "KeyboardTypingEnv"),
                "backend.fly_env.FlyTypingEnv": ("backend.fly_env", "FlyTypingEnv")}


def make_env(config: dict, targets, seed: int | None = None, **overrides):
    """Build the configured environment (point foot or physical fly body)."""
    import importlib
    module, name = ENVIRONMENTS[config.get("environment", "backend.env.KeyboardTypingEnv")]
    cls = getattr(importlib.import_module(module), name)
    env = cls(targets=tuple(targets), gamma=config["ppo"]["gamma"], **{**config["env"], **overrides})
    if seed is not None:
        env.reset(seed=seed)
    return env


def cer(expected: str, actual: str) -> float:
    """Levenshtein distance / target length (raw, case sensitive)."""
    prev = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        cur = [i]
        for j, b in enumerate(actual, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(expected))


def scripted_action(env) -> np.ndarray:
    """Scripted baseline through the env's action interface (its own expert if it has one)."""
    if hasattr(env, "expert_action"):
        return env.expert_action()
    key = physical_key_for(env.target[len(env.buffer)], env.keyboard.shift_latched)
    cx, cy = key_center(key)
    fx, fy, fz = env.foot
    delta = np.array([cx - fx, cy - fy]) / env.step_scale[:2]
    if np.hypot(cx - fx, cy - fy) < 0.15:
        dz = -1.0 if env._armed() else 1.0
    else:
        dz = 1.0 if fz < 0.5 else 0.0
    return np.clip([*delta, dz], -1, 1).astype(np.float32)


def run_episode(env, target: str, seed: int, act) -> dict:
    """Roll one episode; ``act(obs, env)`` returns an action."""
    obs, _ = env.reset(seed=seed, options={"target": target})
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(act(obs, env))
        total += reward
        done = terminated or truncated
    replayed = replay_text(env.events)
    assert replayed == env.typed, "replay reconstruction diverged from env text"
    return {"target": target, "seed": seed, "text": env.typed, "exact": env.typed == target,
            "cer": cer(target, env.typed), "ticks": env.tick, "reward": total,
            "presses": sum(e["type"] == "contact_onset" for e in env.events),
            "wrong_chars": int(info["error"]),
            "wrong_by": next((e["foot"] for e in env.events if e["type"] == "key"
                              and not target.startswith(e.get("text", ""))), None)}


def summarize(episodes: list[dict]) -> dict:
    n = max(1, len(episodes))
    return {"episodes": len(episodes),
            "exact_match": sum(e["exact"] for e in episodes) / n,
            "mean_cer": float(np.mean([e["cer"] for e in episodes])) if episodes else None,
            "mean_ticks": float(np.mean([e["ticks"] for e in episodes])) if episodes else None,
            "mean_reward": float(np.mean([e["reward"] for e in episodes])) if episodes else None}


def eval_targets(tokens: list[str], episodes: int, seed: int) -> list[tuple[str, int]]:
    """Fixed, seed-determined (target, episode_seed) list, cycling through tokens."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tokens))
    return [(tokens[order[i % len(tokens)]], int(seed * 100_000 + i)) for i in range(episodes)]


POLICY_VARIANTS = ("mlp", "connectome", "shuffled", "random_sparse", "dense", "silenced")


def make_model(config: dict, variant: str, seed: int, pool, n_envs: int = 1, vec=None, **kwargs):
    """PPO model with an MLP actor or a connectome-wired actor variant (see training.connectome_policy)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"policy variant must be one of {POLICY_VARIANTS}")
    if vec is None:
        vec = DummyVecEnv([lambda: make_env(config, pool) for _ in range(n_envs)])
    ppo = dict(config["ppo"])
    policy_kwargs = ppo.pop("policy_kwargs", None)
    if variant == "mlp":
        return PPO("MlpPolicy", vec, seed=seed, policy_kwargs=policy_kwargs, **ppo, **kwargs)
    from training.connectome_policy import ConnectomePolicy
    actor = {"mode": "connectome" if variant == "silenced" else variant, "seed": seed,
             "silence": ["proprio"] if variant == "silenced" else []}
    return PPO(ConnectomePolicy, vec, seed=seed, policy_kwargs={"actor_kwargs": actor}, **ppo, **kwargs)
