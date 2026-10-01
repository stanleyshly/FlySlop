"""pass@1 on the micro-suite (greedy) for any policy, per language and per difficulty, plus the repair-loop gain.

    FLYSLOP_MAX_RAM_GB=0.75 uv run --extra training python -m training.eval_microsuite --policy oracle
    ... --policy null | oracle | thinker:<ckpt-dir-or-.pt>   [--mode scratch|corrupt] [--max-rounds 3]

Policies are duck-typed (``act(context_ids) -> token ids``). ``evaluate`` runs every task through
:class:`SelfCorrectEnv`; with ``max_rounds=0`` that is plain pass@1. ``repair_gain`` compares the first run
against the result after up to ``max_rounds`` repair rounds. Judging is the slow part (Verilator ~5 s/task), so
episodes run in threads (subprocess-bound) sized by ``worker_budget``; policy calls are serialised by a lock."""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from backend.memory_budget import worker_budget
from training.microsuite import load_tasks
from training.selfcorrect_env import (JudgeCache, NullPolicy, OracleRepairPolicy, SelfCorrectEnv,
                                      corrupted_start)
from training.tokenizer import BPETokenizer

PER_WORKER_BYTES = 200 << 20      # a Verilator compile+sim subtree peaks well under this; measured in the report
RESERVE_BYTES = 300 << 20


class _Locked:
    def __init__(self, policy):
        self.p, self.lock = policy, threading.Lock()
        if hasattr(policy, "act_env"):
            self.act_env = self._act_env

    def act(self, ctx):
        with self.lock:
            return self.p.act(ctx)

    def _act_env(self, env, ctx):
        with self.lock:
            return self.p.act_env(env, ctx)


def default_workers(max_workers: int = 4) -> int:
    return max(1, worker_budget(PER_WORKER_BYTES, RESERVE_BYTES, max_workers=max_workers))


def default_tokenizer(path: str | Path | None = None) -> BPETokenizer:
    """The trained corpus BPE when present, else a merge-less tokenizer."""
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "data" / "corpus" / "tokenizer" / "bpe.json"
    return BPETokenizer.load(p) if p.is_file() else BPETokenizer()


def evaluate(policy, tasks: Sequence[dict] | None = None, tokenizer: BPETokenizer | None = None, max_rounds: int = 0,
             mode: str = "scratch", n_errors: int = 1, seed: int = 0, workers: int | None = None,
             run_timeout: float = 20.0, judge=None, **env_kw) -> dict:
    """``mode='scratch'``: policy starts from an empty buffer; ``'corrupt'``: from the reference with injected
    errors. Returns per-task rows plus pass@1 (final), first-run pass, and repair gain."""
    tasks = list(tasks if tasks is not None else load_tasks())
    tokenizer = tokenizer or default_tokenizer()
    judge = judge or JudgeCache()
    pol = _Locked(policy)
    workers = workers or default_workers()

    def one(ix_task):
        i, t = ix_task
        env = SelfCorrectEnv(tokenizer, max_rounds=max_rounds, run_timeout=run_timeout, judge=judge, **env_kw)
        start = None
        if mode == "corrupt":
            c = corrupted_start(t, seed + i, n_errors)
            start = c.text
        t0 = time.monotonic()
        info = env.run_episode(pol, t, start_buffer=start, cursor=None)
        info.pop("buffer")
        info.update(lang=t["lang"], difficulty=t["difficulty"], lines=t["lines"], seconds=round(time.monotonic() - t0, 2))
        return info

    t0 = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, enumerate(tasks)))
    wall = time.monotonic() - t0

    def rate(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 4) if rs else None

    groups = defaultdict(list)
    for r in rows:
        groups[f"lang={r['lang']}"].append(r)
        groups[f"difficulty={r['difficulty']}"].append(r)
        groups[f"lang={r['lang']},difficulty={r['difficulty']}"].append(r)
    rep = {"mode": mode, "max_rounds": max_rounds, "n": len(rows), "pass_at_1": rate(rows, "ok"),
           "first_run_pass": rate(rows, "first_ok"),
           "repair_gain": round(rate(rows, "ok") - rate(rows, "first_ok"), 4) if rows else None,
           "mean_repair_rounds": round(sum(r["repair_rounds"] for r in rows) / max(1, len(rows)), 3),
           "by_group": {k: {"n": len(v), "pass_at_1": rate(v, "ok"), "first_run_pass": rate(v, "first_ok")}
                        for k, v in sorted(groups.items())},
           "timing": {"wall_s": round(wall, 1), "workers": workers,
                      "mean_task_s": {l: round(sum(r["seconds"] for r in rows if r["lang"] == l) /
                                               max(1, sum(r["lang"] == l for r in rows)), 2) for l in ("py", "sv")},
                      "judge_cache_hits": getattr(judge, "hits", None), "judge_runs": getattr(judge, "misses", None)},
           "rows": rows}
    return rep


def repair_gain(policy, tasks=None, tokenizer=None, max_rounds: int = 3, mode: str = "scratch", **kw) -> dict:
    """pass@1 after up to ``max_rounds`` repair rounds vs. no repair (same policy, same starts, shared judge cache)."""
    judge = kw.pop("judge", None) or JudgeCache()
    rep = evaluate(policy, tasks, tokenizer, max_rounds=max_rounds, mode=mode, judge=judge, **kw)
    if mode == "corrupt":      # no repair = the corrupted start as given (run once at reset)
        base = {"pass_at_1": rep["first_run_pass"]}
    else:                      # no repair = the policy's single first attempt
        base = evaluate(policy, tasks, tokenizer, max_rounds=0, mode=mode, judge=judge, **kw)
    return {"no_repair_pass_at_1": base["pass_at_1"], "repair_pass_at_1": rep["pass_at_1"],
            "improvement": round(rep["pass_at_1"] - base["pass_at_1"], 4), "max_rounds": max_rounds, "mode": mode,
            "mean_repair_rounds": rep["mean_repair_rounds"], "no_repair": base, "with_repair": rep}


def load_policy(spec: str):
    if spec == "oracle":
        return OracleRepairPolicy()
    if spec == "null":
        return NullPolicy()
    if spec.startswith("thinker:"):
        import torch
        from backend.connectome.thinker import build_model
        from training.selfcorrect_env import ThinkerPolicy
        blob = torch.load(spec.split(":", 1)[1], map_location="cpu")
        model = build_model(blob["kind"], blob.get("cfg"))
        model.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob["model"])
        return ThinkerPolicy(model.eval())
    raise SystemExit(f"unknown policy {spec!r}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="oracle")
    ap.add_argument("--mode", default="scratch", choices=["scratch", "corrupt"])
    ap.add_argument("--max-rounds", type=int, default=0)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="runs/microsuite_report.json")
    a = ap.parse_args(argv)
    tasks = load_tasks(a.lang)[: a.limit]
    rep = evaluate(load_policy(a.policy), tasks, max_rounds=a.max_rounds, mode=a.mode, workers=a.workers)
    rep["policy"] = a.policy
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
