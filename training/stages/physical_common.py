"""Shared helpers for the physical curriculum stages 1-4 (PLAN.md section 3): in-process BC + PPO, held-out
evaluation, warm-start lookup, gate logic. Heavy imports (torch, SB3, mujoco) are deferred so importing a stage
module (and unit-testing the gate logic) stays cheap.

Budget plumbing: every loop ticks ``ctx`` (BC per epoch / episode, PPO per env step, eval per episode), so
``BudgetExceeded`` propagates out of the trainer. Stages catch it, save an ``interrupted`` checkpoint, and re-raise.

Smoke gates: with ``smoke=True`` the gate is *relaxed* (always passes) unless ``stage_cfg["smoke_gates"] == "strict"``;
metrics then carry ``gate_mode = "relaxed (smoke, NOT a real gate)"`` and ``real_gate_passed``.
"""
from __future__ import annotations

import json
import math
from collections import deque
import shutil
import time
from pathlib import Path

RELAXED = "relaxed (smoke, NOT a real gate)"
DEFAULT_PHYS = {"config": "training/ppo_fly_mn.json", "action_mode": "mn", "readout": "mn_antagonist",
                "policy": "connectome", "eval_episodes": 30, "n_envs": 1, "bc_workers": 1,
                "bc_episodes": 400, "bc_epochs": 30, "ppo_timesteps": 200_000, "bc_noise": None,
                "ppo": {}, "max_fit_seconds": None}


# ------------------------------------------------------------------ config / gates (pure, unit-tested)
def _event(ctx, event: str, **fields) -> None:
    """Write run progress to the orchestrator's append-only JSONL log and console."""
    from training.curriculum import append_training_event
    row = {"event": event, "time": time.time(), **fields}
    append_training_event(Path(ctx.run_dir), row, getattr(ctx, "tracker", None))
    print("[training] " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def phys_cfg(stage_cfg: dict, smoke: bool) -> dict:
    """Stage ``physical`` block over DEFAULT_PHYS; in smoke, ``smoke_physical`` is merged on top."""
    out = {**DEFAULT_PHYS, **stage_cfg.get("physical", {})}
    if smoke:
        out.update(stage_cfg.get("smoke_physical", {}))
    return out


def gate_check(metrics: dict, gate: dict, rules: dict) -> tuple[bool, dict]:
    """``rules``: gate key -> (metric name, 'ge'|'le'). Returns (all met, per-key detail). A missing metric fails."""
    detail = {}
    for key, (metric, op) in rules.items():
        want, got = gate.get(key), metrics.get(metric)
        ok = want is not None and got is not None and (got >= want if op == "ge" else got <= want)
        detail[key] = {"metric": metric, "value": got, "threshold": want, "op": op, "ok": bool(ok)}
    return all(d["ok"] for d in detail.values()), detail


def finalize_gate(metrics: dict, real_ok: bool, detail: dict, stage_cfg: dict, smoke: bool) -> bool:
    """Record the gate outcome in ``metrics`` and return what the orchestrator should see."""
    metrics["gate_detail"], metrics["real_gate_passed"] = detail, bool(real_ok)
    if smoke and stage_cfg.get("smoke_gates", "relaxed") != "strict":
        metrics["gate_mode"] = RELAXED
        return True
    metrics["gate_mode"] = "strict"
    return bool(real_ok)


def prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def key_metrics(episodes: list[dict]) -> dict:
    """Stage-1 metrics from single-key episodes: fraction typed correctly, unintended chars per intended press."""
    n = max(1, len(episodes))
    intended = sum(len(e["target"]) for e in episodes)
    unintended = sum(len(e["text"]) - prefix_len(e["target"], e["text"]) for e in episodes)
    return {"key_accuracy": sum(e["text"] == e["target"] for e in episodes) / n,
            "unintended_rate": unintended / max(1, intended), "episodes": len(episodes)}


def exact_rate(episodes: list[dict]) -> float | None:
    return sum(bool(e["exact"]) for e in episodes) / len(episodes) if episodes else None


# ------------------------------------------------------------------ warm start / artifacts
def prev_checkpoint(ctx, stage_id: int) -> str | None:
    """Best checkpoint recorded in the previous stage's artifacts (state.json), else its best_model.zip on disk."""
    state_p = Path(ctx.run_dir) / "state.json"
    if state_p.exists():
        try:
            for e in json.loads(state_p.read_text())["stages"]:
                if e["id"] == stage_id - 1:
                    for a in e.get("artifacts", []):
                        if str(a).endswith("best_model.zip") and Path(a).exists():
                            return str(a)
        except (ValueError, KeyError):
            pass
    cand = Path(ctx.run_dir) / f"stage{stage_id - 1}" / "best_model.zip"
    return str(cand) if cand.exists() else None


def _pick_best(cands: list[tuple[str, float]], dest: Path) -> tuple[str | None, float | None]:
    """Copy the highest-scoring checkpoint to ``dest`` (best_model.zip)."""
    cands = [(p, s) for p, s in cands if p and Path(p).exists()]
    if not cands:
        return None, None
    path, score = max(cands, key=lambda c: c[1])
    if Path(path).resolve() != dest.resolve():
        shutil.copyfile(path, dest)
    return str(dest), score


pick_best = _pick_best


# ------------------------------------------------------------------ physical training (lazy heavy imports)
def build_config(pc: dict) -> dict:
    from training.common import ROOT, load_config
    cfg_path = Path(pc["config"])
    config = load_config(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path)
    config["env"]["action_mode"] = pc["action_mode"]
    if pc["policy"] == "connectome":
        spec = config.setdefault("policy", {"type": "connectome"})
        spec["type"] = "connectome"
        spec.setdefault("connectome", {})["readout"] = pc["readout"]
        if pc.get("circuit"):
            spec["connectome"]["circuit"] = pc["circuit"]
    else:
        config["policy"] = {"type": "mlp"}
    config["ppo"].update(pc.get("ppo", {}))
    config["n_envs"] = pc["n_envs"]
    return config


def make_vec(config: dict, pool, n_envs: int, wrapper=None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from training.common import make_env

    def factory(i):
        def make():
            env = make_env(config, pool)
            return Monitor(wrapper(env, i) if wrapper else env)
        return make
    return (SubprocVecEnv if n_envs > 1 else DummyVecEnv)([factory(i) for i in range(n_envs)])


def new_model(config: dict, pool, seed: int, warm: str | None = None, wrapper=None):
    from training.train_ppo import load_ppo, make_model
    vec = make_vec(config, pool, config["n_envs"], wrapper)
    model = make_model(config, vec, seed=seed, device="cpu")
    if warm:
        model.policy.load_state_dict(load_ppo(warm, device="cpu").policy.state_dict())
    return model


def swap_env(model, config: dict, pool, wrapper=None) -> None:
    old = model.get_env()
    model.set_env(make_vec(config, pool, config["n_envs"], wrapper))
    if old is not None:
        old.close()


def bc_fit(ctx, model, config: dict, obs, act, epochs: int, seed: int, max_seconds=None) -> list[dict]:
    """Behaviour-cloning epochs via ``training.imitate.fit`` (one epoch per call so every epoch ticks the budget)."""
    from training.imitate import fit
    cfg = {**config["imitation"], "epochs": 1, "max_seconds": None}
    details, t0 = [], time.time()
    _event(ctx, "bc_fit_start", stage=Path(ctx.stage_dir).name, epochs=epochs, transitions=len(obs))
    for ep in range(epochs):
        d: list = []
        fit(model, obs, act, cfg, seed + ep, d)
        details += d
        ctx.tick(max(1, len(obs) // max(1, cfg["batch_size"])))
        _event(ctx, "bc_epoch", stage=Path(ctx.stage_dir).name, epoch=ep + 1, epochs=epochs,
               transitions=len(obs), loss=(d[-1].get("total") if d else None), elapsed_s=round(time.time() - t0, 1))
        if max_seconds and time.time() - t0 > max_seconds:
            break
    return details


def bc_collect_pool(ctx, config: dict, pool, episodes: int, seed: int, workers: int):
    """Token/char demonstrations via ``training.imitate.collect`` (process pool; ticks by episodes afterwards)."""
    from training.imitate import collect
    _event(ctx, "bc_collection_start", stage=Path(ctx.stage_dir).name, episodes=episodes, workers=max(1, workers))
    def progress(done, total, transitions, completed_episodes):
        _event(ctx, "bc_collection_progress", stage=Path(ctx.stage_dir).name, workers_done=done,
               workers_total=total, transitions=transitions, episodes=completed_episodes)
    obs, act, stats = collect(config, list(pool), episodes, seed, max(1, workers), progress=progress)
    ctx.tick(int(stats["transitions"]))
    _event(ctx, "bc_collection_end", stage=Path(ctx.stage_dir).name, **stats)
    return obs, act, stats


def ppo_learn(ctx, model, timesteps: int) -> None:
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np

    class Tick(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.monotonic()
            self.last_emit = self.started
            self.initial_steps = int(model.num_timesteps)
            self.last_steps = self.initial_steps
            self.rewards = deque(maxlen=1000)
            self.episodes = deque(maxlen=100)

        def _on_step(self) -> bool:
            ctx.tick(self.training_env.num_envs)
            for reward in self.locals.get("rewards", []):
                self.rewards.append(float(reward))
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep:
                    self.episodes.append({"reward": float(ep["r"]), "length": int(ep["l"])})
            now = time.monotonic()
            if now - self.last_emit >= 5.0 or self.num_timesteps - self.last_steps >= 1000:
                self.emit("ppo_progress", now)
            return True

        def _on_rollout_start(self) -> None:
            # SB3 records optimizer stats after the prior rollout.
            vals = getattr(model.logger, "name_to_value", {})
            metrics = {}
            for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss",
                        "train/approx_kl", "train/clip_fraction", "train/explained_variance"):
                value = vals.get(key)
                if value is not None:
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        metrics[key] = value
            if metrics:
                _event(ctx, "ppo_optimizer", stage=Path(ctx.stage_dir).name,
                       timestep=max(0, self.num_timesteps - self.initial_steps), **metrics)

        def emit(self, event, now):
            span = max(now - self.last_emit, 1e-9)
            phase_steps = max(0, self.num_timesteps - self.initial_steps)
            steps = max(0, self.num_timesteps - self.last_steps)
            vals = getattr(model.logger, "name_to_value", {})
            metrics = {k: float(vals[k]) for k in ("train/loss", "train/policy_gradient_loss", "train/value_loss",
                       "train/entropy_loss", "train/approx_kl", "train/clip_fraction", "train/explained_variance")
                       if k in vals and isinstance(vals[k], (int, float)) and math.isfinite(float(vals[k]))}
            row = {"stage": Path(ctx.stage_dir).name, "timesteps": phase_steps,
                   "total_timesteps": timesteps, "steps_per_s": round(steps / span, 2),
                   "elapsed_s": round(now - self.started, 1), **metrics}
            if self.rewards:
                row["mean_step_reward"] = float(np.mean(self.rewards))
            if self.episodes:
                recent = list(self.episodes)[-20:]
                row["recent_episode_reward"] = float(np.mean([x["reward"] for x in recent]))
                row["recent_episode_length"] = float(np.mean([x["length"] for x in recent]))
                row["recent_episodes"] = len(recent)
            _event(ctx, event, **row)
            self.last_emit, self.last_steps = now, self.num_timesteps

        def _on_training_end(self) -> None:
            self.emit("ppo_end", time.monotonic())

    if timesteps > 0:
        _event(ctx, "ppo_start", stage=Path(ctx.stage_dir).name, total_timesteps=timesteps)
        model.learn(total_timesteps=timesteps, callback=Tick(), reset_num_timesteps=False, progress_bar=False)


def eval_pool(ctx, model, config: dict, targets, episodes: int, seed: int) -> list[dict]:
    """Deterministic held-out rollouts through the env (``run_episode``); one tick per episode."""
    import numpy as np
    from training.common import eval_targets, make_env, run_episode
    env = make_env(config, list(targets))
    policy = lambda o, _e: model.predict(o, deterministic=True)[0]  # noqa: E731
    out = []
    for t, s in eval_targets(list(targets), episodes, seed):
        out.append(run_episode(env, t, s, policy))
        ctx.tick(1)
    if out:
        exact = sum(bool(e.get("exact")) for e in out)
        _event(ctx, "evaluation", stage=Path(ctx.stage_dir).name, episodes=len(out), exact=exact / len(out),
               mean_reward=float(np.mean([e["reward"] for e in out if "reward" in e]))
               if any("reward" in e for e in out) else None)
    return out


def save_model(model, path: Path) -> str:
    model.save(path)
    return str(path)


# ------------------------------------------------------------------ key-queue episodes (stages 3-4)
class KeyQueueWrapper:
    """Gymnasium wrapper (built lazily) that resets the fly env into key-queue mode with sampled key sequences."""

    def __new__(cls, env, sampler, seed: int = 0):
        import gymnasium as gym

        class _W(gym.Wrapper):
            def __init__(self):
                super().__init__(env)
                self.sampler, self.rng_n = sampler, 0
                self.base = seed

            def reset(self, *, seed=None, options=None):
                import random
                self.rng_n += 1
                keys, max_steps = self.sampler(random.Random(self.base * 1_000_003 + self.rng_n), self.env.unwrapped)
                return self.env.reset(seed=seed, options={"keys": keys, "max_steps": max_steps})
        return _W()


class EditSampler:
    """Random typed word + edit-key tail (stage 3) or slipped-then-corrected key strings (stage 4, ``slip_p``>0)."""

    EDIT = ("<BS>", "<LEFT>", "<RIGHT>")
    FN = ("<DEL>", "<HOME>", "<END>")

    def __init__(self, words, include_fn: bool = True, slip_p: float = 0.0, max_edits: int = 3):
        self.words, self.slip_p, self.max_edits = list(words), slip_p, max_edits
        self.tokens = list(self.EDIT + (self.FN if include_fn else ()))

    def commands(self, rng):
        """-> (KeyCommands, expected clean text or None)."""
        from backend.keyplan import expand_tokens, text_to_keys
        word = rng.choice(self.words)
        cmds = text_to_keys(word)
        if self.slip_p > 0:
            from backend.error_inject import corrupt_keys
            bad, slipped = corrupt_keys(cmds, self.slip_p, seed=rng.randrange(2**31))
            if slipped:
                first = slipped[0]
                fix = expand_tokens(["<BS>"] * (len(bad) - first)) + cmds[first:]
                return list(bad) + fix, word
            return cmds, word
        tail = expand_tokens([rng.choice(self.tokens) for _ in range(rng.randint(1, self.max_edits))])
        return cmds + tail + text_to_keys(rng.choice(self.words)[:2]), None

    def compiled(self, rng):
        from training.physical_eval import Unsupported, compile_commands
        for _ in range(20):
            cmds, want = self.commands(rng)
            try:
                keys, _owner = compile_commands(cmds)
            except Unsupported:
                continue
            if keys:
                return cmds, keys, want
        raise RuntimeError("no compilable edit sequence (all Fn-only?)")

    def __call__(self, rng, env):
        _cmds, keys, _want = self.compiled(rng)
        n_press = len(keys) + sum(s for _, s in keys)
        return keys, env.base_steps + env.steps_per_char * n_press


def collect_keys(ctx, config: dict, sampler, episodes: int, seed: int, noise: float):
    """Serial expert demonstrations on key-queue episodes (clean expert action labelled, noisy action executed)."""
    import numpy as np
    from training.common import make_env
    env = make_env(config, ["a"])
    rng, obs_l, act_l, solved = np.random.default_rng(seed), [], [], 0
    import random
    for i in range(episodes):
        episode_steps = 0
        keys, max_steps = sampler(random.Random(seed * 7919 + i), env)
        obs, _ = env.reset(seed=seed * 10_000 + i, options={"keys": keys, "max_steps": max_steps})
        done = False
        while not done:
            expert = env.expert_action()
            obs_l.append(obs)
            act_l.append(expert)
            ex = np.clip(expert + rng.normal(0, noise, expert.shape), -1, 1) if noise else expert
            obs, _r, term, trunc, info = env.step(ex)
            done = term or trunc
            ctx.tick(1)
            episode_steps += 1
        solved += int(env.qi >= len(env.queue) and not info.get("error"))
        if (i + 1) % 10 == 0 or i + 1 == episodes:
            _event(ctx, "key_demo_progress", stage=Path(ctx.stage_dir).name, episode=i + 1, episodes=episodes,
                   episode_steps=episode_steps, expert_complete=solved / (i + 1))
    return np.array(obs_l, np.float32), np.array(act_l, np.float32), {"episodes": episodes, "transitions": len(obs_l),
                                                                      "expert_complete": solved / max(1, episodes)}


def eval_edit_keys(ctx, model, config: dict, sampler: EditSampler, episodes: int, seed: int) -> dict:
    """Physical edit-key accuracy: commands typed correctly first try (``physical_eval.type_tokens``), held-out seeds.

    Held-mode chords the typist cannot realise come back ``unsupported`` and are excluded from the accuracy (counted)."""
    import random
    from training.common import make_env
    from training.physical_eval import type_tokens
    env = make_env(config, ["a"], terminate_on_error=False)
    n_ok = n_cmd = unsupported = exact = 0
    for i in range(episodes):
        cmds, _keys, _ = sampler.compiled(random.Random(10**6 + seed * 997 + i))
        res = type_tokens(cmds, "expert", config["env"]["action_mode"], seed=seed + i, recover=False, env=env, model=model)
        if res.get("unsupported"):
            unsupported += 1
        else:
            n_ok += res["success_rate"] * res["n_commands"]
            n_cmd += res["n_commands"]
            exact += int(bool(res["exact"]))
        ctx.tick(1)
        if (i + 1) % 10 == 0 or i + 1 == episodes:
            _event(ctx, "evaluation_progress", stage=Path(ctx.stage_dir).name, kind="edit_keys",
                   episodes=i + 1, total_episodes=episodes,
                   accuracy=(n_ok / n_cmd) if n_cmd else None, unsupported=unsupported)
    result = {"edit_key_accuracy": (n_ok / n_cmd) if n_cmd else None, "edit_exact": exact / max(1, episodes - unsupported),
              "edit_episodes": episodes, "edit_unsupported": unsupported}
    _event(ctx, "evaluation", stage=Path(ctx.stage_dir).name, kind="edit_keys", **result)
    return result


def eval_recovery(ctx, model, config: dict, words, episodes: int, seed: int, noise: float) -> dict:
    """Stage-4 physical gate: type held-out words with motor noise + replanning recovery (``type_tokens``)."""
    from training.common import eval_targets, make_env
    from training.physical_eval import type_tokens
    env = make_env(config, ["a"], terminate_on_error=False)
    exact = slipped = rec = unsupported = 0
    for i, (w, s) in enumerate(eval_targets(list(words), episodes, seed)):
        res = type_tokens(w, "expert", config["env"]["action_mode"], seed=s, recover=True, env=env, model=model, noise=noise)
        unsupported += int(bool(res.get("unsupported")))
        exact += int(bool(res["exact"]))
        if res["slips"]:
            slipped += 1
            rec += int(bool(res["exact"]))
        ctx.tick(1)
        if (i + 1) % 10 == 0 or i + 1 == episodes:
            _event(ctx, "evaluation_progress", stage=Path(ctx.stage_dir).name, kind="recovery",
                   episodes=i + 1, total_episodes=episodes, exact=exact / (i + 1), slipped=slipped,
                   recovered_given_slip=(rec / slipped) if slipped else None, unsupported=unsupported)
    result = {"recovery_exact": exact / max(1, episodes), "slip_episodes": slipped,
              "recovered_given_slip": (rec / slipped) if slipped else None, "recovery_episodes": episodes,
              "recovery_noise": noise, "recovery_unsupported": unsupported}
    _event(ctx, "evaluation", stage=Path(ctx.stage_dir).name, kind="recovery", **result)
    return result


class SubCtx:
    """Proxy so a symbolic trainer writes into its own sub-directory while sharing the parent's budget."""

    def __init__(self, ctx, sub: str):
        self._ctx = ctx
        self.stage_dir = Path(ctx.stage_dir) / sub
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def run_symbolic(ctx, stage_cfg: dict, smoke: bool, stage: int) -> dict:
    """Lazy-import ``training.edit_trainer`` and run its stage; ``{"status": "symbolic part pending"}`` if missing."""
    try:
        from training import edit_trainer
    except ImportError as e:
        return {"status": "symbolic part pending", "detail": str(e), "gate_passed": False}
    sc = dict(stage_cfg)
    ov = dict(sc.get("overrides", {}))
    if "wall_s" in ctx.budget:
        ov.setdefault("train.max_seconds", max(5.0, ctx.budget["wall_s"] - ctx.elapsed - 5.0))
    sc["overrides"] = ov
    res = edit_trainer.run_stage(SubCtx(ctx, "symbolic"), sc, smoke, stage=stage)
    return {"status": "ok", "metrics": res.metrics, "gate_passed": bool(res.gate_passed), "artifacts": list(res.artifacts)}


def train_phases(ctx, config: dict, pc: dict, phases: list[dict], warm: str | None, seed: int, out_dir: Path,
                 on_phase=None) -> tuple[list[dict], list[tuple[str, float]], str | None]:
    """BC -> PPO over ``phases`` (each ``{name, pool, bc_episodes, bc_epochs, ppo_timesteps, collect?, wrapper?}``),
    warm-started from ``warm``. A phase checkpoint is written after each phase; ``on_phase(name, model)`` may return
    a score used to choose the best checkpoint. On BudgetExceeded an ``interrupted.zip`` is saved and it re-raises.
    Returns (log, [(ckpt, score)], last checkpoint)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log, cands, last, model = [], [], None, None
    if warm:
        cands.append((warm, -1.0))          # the warm start itself competes (score -1 unless evaluated)
    try:
        for i, ph in enumerate(phases):
            rec = {"phase": ph["name"], "pool": len(ph["pool"]) if hasattr(ph["pool"], "__len__") else None}
            _event(ctx, "phase_start", phase=ph["name"], phase_index=i + 1, phases=len(phases), pool=rec["pool"])
            wrapper = ph.get("wrapper")
            if model is None:
                model = new_model(config, ph["pool"], seed + i, warm, wrapper)
                rec["warm_start"] = warm
            else:
                swap_env(model, config, ph["pool"], wrapper)
            t0 = time.time()
            if ph.get("bc_episodes", 0) and ph.get("bc_epochs", 0):
                if ph.get("collect"):
                    _event(ctx, "bc_collection_start", phase=ph["name"], episodes=ph["bc_episodes"])
                    obs, act, stats = ph["collect"](ctx, config, ph["bc_episodes"], seed + 31 * i)
                    _event(ctx, "bc_collection_end", phase=ph["name"], **stats)
                else:
                    obs, act, stats = bc_collect_pool(ctx, config, ph["pool"], ph["bc_episodes"], seed + 31 * i,
                                                      min(pc["bc_workers"], max(1, ctx.max_workers)))
                d = bc_fit(ctx, model, config, obs, act, ph["bc_epochs"], seed + i, pc.get("max_fit_seconds"))
                rec["bc"] = {**stats, "epochs": len(d), "mse_first": d[0]["total"] if d else None,
                             "mse_last": d[-1]["total"] if d else None}
                del obs, act
            if ph.get("ppo_timesteps", 0):
                ppo_learn(ctx, model, ph["ppo_timesteps"])
                rec["ppo_timesteps"] = ph["ppo_timesteps"]
            last = save_model(model, out_dir / f"phase{i}_{ph['name']}.zip")
            rec["seconds"] = round(time.time() - t0, 1)
            rec["ckpt"] = last
            if on_phase:
                score = on_phase(ph["name"], model, rec)
                if score is not None:
                    cands.append((last, score))
            log.append(rec)
    except BaseException:
        if model is not None:
            save_model(model, out_dir / "interrupted.zip")
        raise
    finally:
        if model is not None and model.get_env() is not None:
            model.get_env().close()
    return log, cands, last
