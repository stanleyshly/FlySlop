"""Stage 9 self-correction fine-tuner (PLAN.md section 3) plus checkpoint / policy adapters shared by stages 6-9.

Two thinker models cooperate:

* ``G`` (generation policy, the stage 6/8 distilled model): ``BOS prompt SEP -> code EOS``.
* ``R`` (repair policy): starts from the stage 3-4 edit policy weights and is fine-tuned here on the
  :class:`training.selfcorrect_env.SelfCorrectEnv` context ``BOS #TASK prompt #BUF buffer #OUT output``:
  the sequence is ``context SEP actions EOS`` with loss on ``actions + EOS``; at act time a trailing EOS is
  replaced by ``<RUN>`` so every round ends in a judged run.

Round 0 of an episode that starts from an empty buffer is written by ``G`` (greedy) and run; every later round
(and every round of an episode that starts from a corrupted buffer) is ``R``.

Training data comes from TRAIN tasks only: dataset records that carry tests (MBPP, HumanEval, VerilogEval, ...),
started either from ``backend.error_inject.corrupt_buffer(reference)`` or from ``G``'s own failing attempt.
The held-out micro-suite (``training.microsuite``) is used for evaluation only; ``assert_not_heldout`` enforces it.

Methods (``--method``):
* ``rft``       rejection-sampling fine-tuning: keep the last round of every successful repair, teacher-forced SFT.
* ``reinforce`` REINFORCE with a baseline (``batch`` mean, per-task leave-one-out ``loo`` or running ``ema``),
                reward = 1 iff the final buffer passes its tests.

Full run (offline; the curriculum stage 9 calls the same code):
    FLYSLOP_MAX_RAM_GB=4 uv run --extra training python -m training.selfcorrect_train \
        --gen runs/curriculum/<run>/stage8/ckpt.pt --repair runs/curriculum/<run>/stage4/symbolic/ckpt.pt \
        --method rft --iters 40 --tasks-per-iter 32 --samples 4 --out runs/selfcorrect/rft
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.connectome.thinker import BOS, EOS, PAD, SEP, ThinkerConfig, build_model  # noqa: E402
from backend.error_inject import corrupt_buffer  # noqa: E402
from training import thinker_train_common as C  # noqa: E402
from training.selfcorrect_env import JudgeCache, SelfCorrectEnv  # noqa: E402
from training.tokenizer import RUN, BPETokenizer, normalize  # noqa: E402

DEFAULT_SC = {
    "method": "rft", "iters": 40, "tasks_per_iter": 32, "samples": 4, "temperature": 1.0, "ft_steps": 8,
    "batch": 16, "lr_scale": 0.3, "max_rounds": 3, "n_errors": [1, 2], "scratch_frac": 0.3, "max_ctx": 512,
    "max_new_gen": 256, "max_new_edit": 96, "run_timeout": 20.0, "lang": "py", "max_code_chars": 600,
    "keep_rounds": "last", "baseline": "loo", "ema": 0.9, "n_train_tasks": 400, "verify_reference": True,
    "train_seconds": None, "workers": None, "warmup_examples": 256, "warmup_per_task": 2, "warmup_steps": 100,
    "warmup_lr_scale": 1.0,
}


# ------------------------------------------------------------------ checkpoints
def load_thinker(path, map_location: str = "cpu"):
    """Load a thinker checkpoint. Accepts ``Trainer.save`` / ``distill.py`` files (``model`` + ``kind`` + ``cfg``) and
    the ``state_dict`` spelling that ``eval_microsuite.load_policy`` assumes. Returns ``(model.eval(), blob)``."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    sd = blob.get("model", blob.get("state_dict")) if isinstance(blob, dict) else None
    if sd is None or "kind" not in blob:
        raise ValueError(f"{path}: not a thinker checkpoint (keys {list(blob)[:8] if isinstance(blob, dict) else type(blob)})")
    names = {f.name for f in fields(ThinkerConfig)}
    cfg = ThinkerConfig(**{k: v for k, v in (blob.get("cfg") or {}).items() if k in names})
    model = build_model(blob["kind"], cfg)
    model.load_state_dict(sd)
    return model.eval(), blob


def save_thinker(model, path, opt=None, step: int = 0, extra: dict | None = None) -> str:
    """Same layout as ``Trainer.save`` so ``load_thinker`` / ``Trainer.load`` / ``eval_microsuite`` all read it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt is not None else None, "step": step,
                "elapsed": 0.0, "history": [], "seed": model.cfg.seed, "kind": model.kind, "cfg": model.cfg.__dict__,
                "extra": extra or {}}, tmp)
    tmp.replace(path)
    return str(path)


def find_stage_ckpt(ctx, stage_id: int, name: str = "ckpt.pt") -> str | None:
    """Checkpoint a previous stage of THIS run produced: recorded artifacts first, then the conventional paths."""
    run_dir = Path(ctx.run_dir)
    try:
        for e in json.loads((run_dir / "state.json").read_text())["stages"]:
            if e["id"] == stage_id:
                for a in e.get("artifacts", []):
                    if str(a).endswith(name) and Path(a).exists():
                        return str(a)
    except (OSError, ValueError, KeyError):
        pass
    for rel in (f"stage{stage_id}/{name}", f"stage{stage_id}/symbolic/{name}"):
        if (run_dir / rel).exists():
            return str(run_dir / rel)
    return None


def find_edit_ckpt(ctx) -> str | None:
    """Stage 3-4 symbolic edit policy: this run (stage 4, then 3), then the newest one under the runs directory."""
    for sid in (4, 3):
        p = find_stage_ckpt(ctx, sid)
        if p:
            return p
    cands = []
    for base in {Path(ctx.run_dir).parent, ROOT / "runs" / "curriculum"}:
        for sid in (4, 3):
            cands += list(base.glob(f"*/stage{sid}/symbolic/ckpt.pt"))
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None


# ------------------------------------------------------------------ tasks (train side)
def heldout_keys() -> tuple[set, set]:
    from training.microsuite import load_tasks
    ts = load_tasks(max_difficulty=99)
    return {t["id"] for t in ts}, {normalize(t["prompt"]).strip() for t in ts}


def assert_not_heldout(tasks: list[dict]) -> None:
    ids, prompts = heldout_keys()
    bad = [t["id"] for t in tasks if t["id"] in ids or normalize(t["prompt"]).strip() in prompts]
    if bad:
        raise AssertionError(f"held-out micro-suite tasks leaked into training tasks: {bad[:5]}")


def build_train_tasks(records: list[dict], lang: str = "py", n: int = 400, seed: int = 0, max_code_chars: int = 600,
                      judge=None, verify: bool = False, timeout: float = 20.0) -> list[dict]:
    """Dataset records with tests -> SelfCorrectEnv tasks (id, lang, prompt, code, tests), deterministic in ``seed``.
    ``verify`` keeps only records whose reference passes its own tests (so a repair can succeed)."""
    rs = [r for r in records if r["lang"] == lang and r.get("tests") and len(r["code"]) <= max_code_chars]
    random.Random(seed).shuffle(rs)
    out = []
    for r in rs:
        t = {"id": r["id"], "lang": r["lang"], "prompt": normalize(r["prompt"]), "code": normalize(r["code"]),
             "tests": r["tests"], "source": r.get("source")}
        if verify:
            if not (judge or JudgeCache())(t["lang"], t["code"], t["tests"], timeout)["ok"]:
                continue
        out.append(t)
        if len(out) >= n:
            break
    assert_not_heldout(out)
    return out


# ------------------------------------------------------------------ policies
def _cut(row: list[int]) -> list[int]:
    row = row[:row.index(EOS)] if EOS in row else row
    return [t for t in row if t != PAD]


class SelfCorrectPolicy:
    """``act_env(env, ctx)`` for :class:`SelfCorrectEnv` (duck-typed). ``gen`` writes round 0 from an empty buffer
    (greedy), ``repair`` handles the rest. With ``gen`` only (``repair=None``) later rounds emit ``<EOS>``; with
    ``repair`` only, round 0 also goes through ``repair``. Sampled repair outputs are appended to ``env._trace`` as
    ``(context_ids, action_ids)`` so rollouts can be trained on."""

    def __init__(self, gen, repair, tok, max_new_gen: int = 256, max_new_edit: int = 96, temperature: float = 0.0,
                 max_prompt: int = 384):
        self.gen, self.repair, self.tok = gen, repair, tok
        self.max_new_gen, self.max_new_edit, self.temperature, self.max_prompt = max_new_gen, max_new_edit, temperature, max_prompt
        self.lock = threading.Lock()

    def _generate(self, model, prompt: list[int], max_new: int, temperature: float) -> list[int]:
        with self.lock, torch.no_grad():
            out = model.generate([prompt], max_new=max_new, temperature=temperature)
        return _cut(out[0].tolist())

    def write(self, task: dict) -> list[int]:
        prompt = [BOS] + self.tok.encode(normalize(task["prompt"]))[-self.max_prompt:]
        return self._generate(self.gen, prompt, self.max_new_gen, 0.0)

    def act_env(self, env, ctx):
        if self.gen is not None and env.rounds == 0 and env.editor.text == "":
            return self.write(env.task) + [RUN]
        if self.repair is None:
            return [EOS]
        ids = self._generate(self.repair, list(ctx), self.max_new_edit, self.temperature)
        if hasattr(env, "_trace"):
            env._trace.append((list(ctx), ids))
        return ids + [RUN]

    def act(self, ctx):
        raise RuntimeError("SelfCorrectPolicy needs the env (use env.run_episode)")


class GenPolicy(SelfCorrectPolicy):
    """Stage 7/8 policy: write once from the prompt (``BOS prompt SEP -> code``, the distillation format), then run."""

    def __init__(self, model, tok, max_new: int = 256):
        super().__init__(model, None, tok, max_new_gen=max_new)


# ------------------------------------------------------------------ rollouts
def make_start(task: dict, k: int, seed: int, n_errors: list[int], scratch_frac: float) -> str | None:
    """Start buffer for sample ``k`` of ``task``: None (G writes it) with probability ``scratch_frac``, else the
    reference with ``n_errors[k % len]`` injected errors (``error_inject``, seeded by task id, seed and k)."""
    rng = random.Random(f"{task['id']}|{seed}|{k}")
    if rng.random() < scratch_frac:
        return None
    return corrupt_buffer(task["code"], seed=rng.randrange(1 << 30), n=n_errors[k % len(n_errors)]).text


def rollout(policy, task: dict, start: str | None, tok, sc: dict, judge) -> dict:
    env = SelfCorrectEnv(tok, max_rounds=sc["max_rounds"], max_tokens_per_round=sc["max_new_edit"] + 8,
                         run_timeout=sc["run_timeout"], max_ctx=sc["max_ctx"], judge=judge)
    env._trace = []
    ctx = env.reset(task, start)
    skipped = bool(start is not None and env.runs and env.runs[0]["ok"])   # nothing to repair
    while not env.done and not skipped:
        toks = policy.act_env(env, ctx)
        ctx, _, _, _ = env.step(toks)
    return {"task": task["id"], "reward": env.reward, "skipped": skipped, "trace": list(env._trace),
            "first_ok": bool(env.runs[0]["ok"]) if env.runs else False, "rounds": env.rounds,
            "from_scratch": start is None}


def collect_rollouts(policy, tasks: list[dict], tok, sc: dict, judge, seed: int, workers: int = 1) -> list[dict]:
    jobs = [(t, make_start(t, k, seed, sc["n_errors"], sc["scratch_frac"])) for t in tasks for k in range(sc["samples"])]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(lambda j: rollout(policy, j[0], j[1], tok, sc, judge), jobs))


def make_example(ctx_ids: list[int], act_ids: list[int], max_len: int):
    """``context SEP actions EOS`` with loss on actions + EOS; None if it does not fit ``max_len``."""
    ids = list(ctx_ids) + [SEP] + list(act_ids) + [EOS]
    if len(ids) > max_len:
        return None
    n_out = len(act_ids) + 1
    return ids, [False] * (len(ids) - n_out) + [True] * n_out


def rft_examples(rollouts: list[dict], max_len: int, keep_rounds: str = "last") -> list[tuple]:
    """Successful repair trajectories -> teacher-forcing examples (deduplicated)."""
    seen, out = set(), []
    for r in rollouts:
        if r["skipped"] or r["reward"] <= 0 or not r["trace"]:
            continue
        for ctx_ids, act in (r["trace"][-1:] if keep_rounds == "last" else r["trace"]):
            key = (tuple(ctx_ids), tuple(act))
            ex = None if key in seen else make_example(ctx_ids, act, max_len)
            if ex is not None:
                seen.add(key)
                out.append(ex)
    return out


def oracle_examples(tasks: list[dict], tok, sc: dict, max_len: int, seed: int, judge=None) -> list[tuple]:
    """Supervised warm-up data in the stage 9 context format: corrupted train buffers -> ``oracle_edits`` -> reference.
    (The stage 3-4 edit policy was trained on ``target SEP buffer`` states; this teaches ``R`` the ``#TASK/#BUF/#OUT``
    context before rejection sampling starts.)"""
    from backend.keyplan import oracle_edits
    from training.edit_trainer import enc_actions
    judge, out = judge or JudgeCache(), []
    for t in tasks:
        for k in range(sc["warmup_per_task"]):
            start = make_start(t, k, seed, sc["n_errors"], 0.0)
            env = SelfCorrectEnv(tok, max_rounds=sc["max_rounds"], max_ctx=sc["max_ctx"], run_timeout=sc["run_timeout"], judge=judge)
            ctx = env.reset(t, start)
            if env.runs[0]["ok"]:       # the observation says PASS: the right action is to change nothing (just run)
                toks = []
            else:
                toks = oracle_edits(env.editor.text, env.editor.cursor, env.editor.selection, t["code"])
            ex = make_example(ctx, enc_actions(tok, toks), max_len)
            if ex is not None:
                out.append(ex)
    return out


def advantages(rewards: list[float], groups: list, mode: str = "batch", state: dict | None = None, ema: float = 0.9):
    """REINFORCE baselines: ``batch`` mean, ``loo`` per-group leave-one-out (falls back to batch mean for singleton
    groups) or ``ema`` running mean (``state["b"]`` persists across calls)."""
    n = len(rewards)
    if not n:
        return []
    mean = sum(rewards) / n
    if mode == "ema":
        state = state if state is not None else {}
        b = state.get("b", mean)
        out = [r - b for r in rewards]
        state["b"] = ema * b + (1 - ema) * mean
        return out
    if mode == "loo":
        by = {}
        for i, g in enumerate(groups):
            by.setdefault(g, []).append(i)
        out = []
        for i, g in enumerate(groups):
            idx = by[g]
            out.append(rewards[i] - (sum(rewards[j] for j in idx if j != i) / (len(idx) - 1) if len(idx) > 1 else mean))
        return out
    return [r - mean for r in rewards]


# ------------------------------------------------------------------ updates
def _chunk(model, tcfg: dict) -> int:
    return tcfg.get("chunk") or model.cfg.tbptt or 32


def sft(model, opt, items: list[tuple], steps: int, batch: int, tcfg: dict, seed: int = 0, ctx=None, log=print) -> dict:
    """``steps`` teacher-forced optimiser steps on ``items`` (deterministic batch order in (seed, step))."""
    if not items or steps <= 0:
        return {"steps": 0}
    gen = torch.Generator().manual_seed(seed)
    model.train()
    loss = acc = 0.0
    for s in range(steps):
        idx = C.sample_indices(len(items), batch, seed, s)
        seq, mask = C.pad_batch([items[i] for i in idx])
        loss, acc, _, _ = C.train_step(model, opt, seq, mask, 0.0, gen, _chunk(model, tcfg), 1.0, tcfg.get("clip", 1.0))
        if ctx is not None:
            ctx.tick(1)
    model.eval()
    return {"steps": steps, "loss": round(float(loss), 4), "acc": round(float(acc), 4), "n": len(items)}


def pg_step(model, opt, items: list[tuple], advs: list[float], tcfg: dict) -> float:
    """One REINFORCE step: minimise ``mean_tokens(A_i * -log p(action tokens of sample i))`` (backward per chunk)."""
    seq, mask = C.pad_batch(items)
    x, y, m = seq[:, :-1], seq[:, 1:], mask[:, 1:]
    n = max(int(m.sum()), 1)
    w = torch.tensor(advs, dtype=torch.float32)
    opt.zero_grad(set_to_none=True)
    total = 0.0
    model.train()
    for t0, logits in model.iter_chunks(x, chunk=_chunk(model, tcfg)):
        c = logits.shape[1]
        yc, mc = y[:, t0:t0 + c], m[:, t0:t0 + c]
        if not mc.any():
            continue
        rows = mc.nonzero()[:, 0]
        ce = nn.functional.cross_entropy(logits[mc], yc[mc], reduction="none")
        loss = (ce * w[rows]).sum() / n
        loss.backward()
        total += float(loss.detach())
    if tcfg.get("clip"):
        nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], tcfg["clip"])
    opt.step()
    model.eval()
    return total


# ------------------------------------------------------------------ evaluation
def eval_repair(gen, repair, tok, tasks: list[dict], sc: dict, mode: str, workers: int, seed: int = 0) -> dict:
    """pass@1 after up to ``max_rounds`` repair rounds vs no repair (``eval_microsuite.repair_gain``), greedy."""
    from training import eval_microsuite as em
    pol = SelfCorrectPolicy(gen, repair, tok, sc["max_new_gen"], sc["max_new_edit"], 0.0)
    rg = em.repair_gain(pol, tasks, tok, max_rounds=sc["max_rounds"], mode=mode, seed=seed, workers=workers,
                        run_timeout=sc["run_timeout"], max_ctx=sc["max_ctx"], max_tokens_per_round=sc["max_new_edit"] + 8)

    def by_lang(rep):
        return {k.split("=", 1)[1]: v["pass_at_1"] for k, v in rep.get("by_group", {}).items() if k.count("=") == 1 and k.startswith("lang=")}
    return {"mode": mode, "n": rg["with_repair"]["n"], "no_repair_pass_at_1": rg["no_repair_pass_at_1"],
            "repair_pass_at_1": rg["repair_pass_at_1"], "improvement": rg["improvement"], "max_rounds": rg["max_rounds"],
            "mean_repair_rounds": rg["mean_repair_rounds"], "no_repair_by_lang": by_lang(rg["no_repair"]),
            "repair_by_lang": by_lang(rg["with_repair"])}


# ------------------------------------------------------------------ training loop
def train_selfcorrect(gen, repair, tok, tasks: list[dict], tcfg: dict, sc: dict, workers: int = 1, seed: int = 0,
                      ctx=None, judge=None, log=print) -> dict:
    """``iters`` x (roll out ``tasks_per_iter`` train tasks x ``samples`` starts with sampled repairs, then update
    ``repair`` by RFT or REINFORCE). ``ctx.tick`` per iteration (BudgetExceeded propagates; caller saves)."""
    judge = judge or JudgeCache()
    lr = tcfg["lr"] * sc["lr_scale"]
    opt = C.make_optimizer(repair, lr, tcfg["conn_lr"] * sc["lr_scale"])
    for g in opt.param_groups:
        g["lr"] = g["base"]
    max_len = tcfg["max_len"]
    policy = SelfCorrectPolicy(gen, repair, tok, sc["max_new_gen"], sc["max_new_edit"], sc["temperature"])
    rng = random.Random(seed)
    hist, base_state, t0 = [], {}, time.time()
    for it in range(sc["iters"]):
        if sc.get("train_seconds") and time.time() - t0 > sc["train_seconds"]:
            log(f"[selfcorrect] train_seconds reached at iter {it}")
            break
        batch_tasks = rng.sample(tasks, min(sc["tasks_per_iter"], len(tasks)))
        ro = collect_rollouts(policy, batch_tasks, tok, sc, judge, seed * 1000 + it, workers)
        live = [r for r in ro if not r["skipped"]]
        n_ok = sum(r["reward"] > 0 for r in live)
        row = {"iter": it, "rollouts": len(ro), "repairable": len(live), "solved": n_ok,
               "solve_rate": round(n_ok / max(len(live), 1), 4), "method": sc["method"]}
        if sc["method"] == "reinforce":
            items, rewards, groups = [], [], []
            for r in live:
                for c, a in (r["trace"] if sc["keep_rounds"] == "all" else r["trace"][-1:]):
                    ex = make_example(c, a, max_len)
                    if ex is not None:
                        items.append(ex); rewards.append(r["reward"]); groups.append(r["task"])
            advs = advantages(rewards, groups, sc["baseline"], base_state, sc["ema"])
            loss, done = 0.0, 0
            for b0 in range(0, len(items), sc["batch"]):
                if any(advs[b0:b0 + sc["batch"]]):
                    loss += pg_step(repair, opt, items[b0:b0 + sc["batch"]], advs[b0:b0 + sc["batch"]], tcfg)
                    done += 1
            row.update({"examples": len(items), "updates": done, "loss": round(loss / max(done, 1), 5),
                        "mean_reward": round(sum(rewards) / max(len(rewards), 1), 4)})
            if ctx is not None:
                ctx.tick(1)
        else:
            ex = rft_examples(ro, max_len, sc["keep_rounds"])
            res = sft(repair, opt, ex, sc["ft_steps"], sc["batch"], tcfg, seed * 1000 + it, ctx, log)
            row.update({"examples": len(ex), **{k: v for k, v in res.items() if k in ("loss", "acc", "steps")}})
            if not ex and ctx is not None:
                ctx.tick(1)
        hist.append(row)
        log(f"[selfcorrect] {json.dumps(row)}")
        tracker = getattr(ctx, "tracker", None)
        if tracker is not None:
            tracker.log_values(f"curriculum/{Path(ctx.stage_dir).name}/selfcorrect/{sc['method']}", row)
    return {"history": hist, "seconds": round(time.time() - t0, 1), "opt": opt, "judge_runs": getattr(judge, "misses", None)}


def run_selfcorrect(gen, repair, tok, tcfg: dict, sc: dict, out_dir, train_tasks: list[dict], eval_tasks: list[dict],
                    workers: int = 1, seed: int = 0, ctx=None, log=print) -> dict:
    """Evaluate before, fine-tune ``repair``, evaluate after. Saves ``repair_ckpt.pt`` (also on BudgetExceeded)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert_not_heldout(train_tasks)
    modes = sc.get("eval_modes", ["scratch", "corrupt"])
    before = {m: eval_repair(gen, repair, tok, eval_tasks, sc, m, workers, seed) for m in modes}
    log(f"[selfcorrect] before: {json.dumps({m: (v['no_repair_pass_at_1'], v['repair_pass_at_1']) for m, v in before.items()})}")
    tr, opt, ckpt, warm = {"history": []}, None, str(out / "repair_ckpt.pt"), {}
    try:
        if sc["warmup_examples"] and sc["warmup_steps"]:
            items = oracle_examples(train_tasks[:sc["warmup_examples"]], tok, sc, tcfg["max_len"], seed)
            wopt = C.make_optimizer(repair, tcfg["lr"] * sc["warmup_lr_scale"], tcfg["conn_lr"] * sc["warmup_lr_scale"])
            for g in wopt.param_groups:
                g["lr"] = g["base"]
            warm = sft(repair, wopt, items, sc["warmup_steps"], sc["batch"], tcfg, seed, ctx, log)
            log(f"[selfcorrect] oracle warm-up {json.dumps(warm)}")
        tr = train_selfcorrect(gen, repair, tok, train_tasks, tcfg, sc, workers, seed, ctx, log=log)
        opt = tr.pop("opt")
    finally:
        save_thinker(repair, ckpt, opt, extra={"stage9": {"method": sc["method"], "iters": len(tr["history"])}})
    after = {m: eval_repair(gen, repair, tok, eval_tasks, sc, m, workers, seed) for m in modes}
    return {"before": before, "after": after, "train": tr, "warmup": warm, "checkpoint": ckpt}


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="generation policy checkpoint (stage 6/8 ckpt.pt)")
    ap.add_argument("--repair", default=None, help="repair init checkpoint (stage 3-4 edit ckpt); default: copy of --gen")
    ap.add_argument("--out", default="runs/selfcorrect/run")
    ap.add_argument("--config", default=None, help="thinker.json path")
    ap.add_argument("--method", choices=["rft", "reinforce"], default="rft")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--tasks-per-iter", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--lang", default="py")
    ap.add_argument("--eval-limit-py", type=int, default=None)
    ap.add_argument("--eval-limit-sv", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ram-gb", type=float, default=None)
    a = ap.parse_args(argv)
    from backend.memory_budget import worker_budget
    from training.distill import _tok, load_pairs
    from training.microsuite import load_tasks
    C.guard_ram(a.ram_gb)
    cfg = C.load_cfg(a.config)
    sc = {**DEFAULT_SC, "method": a.method, "lang": a.lang}
    for k, v in (("iters", a.iters), ("tasks_per_iter", a.tasks_per_iter), ("samples", a.samples)):
        if v is not None:
            sc[k] = v
    gen, _ = load_thinker(a.gen)
    repair, _ = load_thinker(a.repair or a.gen)
    tok = _tok(cfg)
    workers = a.workers or max(1, worker_budget(200 << 20, 300 << 20, config_gb=a.ram_gb, max_workers=4))
    recs = load_pairs("dataset", seed=a.seed)["train"]
    train_tasks = build_train_tasks(recs, sc["lang"], sc["n_train_tasks"], a.seed, sc["max_code_chars"],
                                    verify=sc["verify_reference"])
    ev = (load_tasks("py")[:a.eval_limit_py] if a.eval_limit_py is not None else load_tasks("py")) + \
         (load_tasks("sv")[:a.eval_limit_sv] if a.eval_limit_sv is not None else load_tasks("sv"))
    tcfg = {**cfg["train"], "max_len": sc["max_ctx"] + sc["max_new_edit"] + 2}
    res = run_selfcorrect(gen, repair, tok, tcfg, sc, a.out, train_tasks, ev, workers, a.seed)
    res["train"].pop("opt", None)
    Path(a.out, "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({m: {k: v for k, v in r.items() if k.endswith("pass_at_1") or k == "improvement"}
                      for m, r in res["after"].items()}, indent=1))


if __name__ == "__main__":
    main()
