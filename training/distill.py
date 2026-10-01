"""Stage 6 distillation (P6b): teach the thinker (prompt -> code) from oracle / dataset pairs.

Sequence: ``BOS prompt SEP code EOS``; loss on code + EOS only.  Schedule A (teacher forcing) -> B (scheduled
sampling, 25/50/75/~95% of code inputs replaced by the student's own samples) -> C (student-only), logged
per step in ``<out>/schedule.jsonl`` together with ``"event": "transition"`` rows (reason threshold / step_cap).
``schedule.mode`` = ``gated`` (default; advance on rolling accuracy, step-fraction caps) or ``fixed``;
``train.mix_mode`` = ``sequential`` (free-run) or ``parallel`` (Bernoulli, one pass); ``train.tf_lambda`` keeps a
teacher-forced loss in B/C.  Metrics: held-out token accuracy and greedy-generation parse/exec
pass rate via ``backend.exec_judge`` on a capped sample.

Full runs (offline, the user runs these):
    uv run --extra training python -m training.distill --kind real --steps 20000 --out runs/distill/real
    uv run --extra training python -m training.distill --kind real --resume runs/distill/real/ckpt.pt --steps 20000 --out runs/distill/real
    uv run --extra training python -m training.distill --controls --seeds 0 1 2 3 4 --steps 20000 --out runs/distill/controls
    uv run --extra training python -m training.distill --overfit 100 --kind real --steps 1500 --out runs/distill/overfit_real
Set FLYSLOP_MAX_RAM_GB (default 4).  Speed knobs: --k-steps (2 reaches every DN), --max-len, --batch, --tbptt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import exec_judge  # noqa: E402
from backend.connectome.thinker import BOS, EOS, KINDS, SEP  # noqa: E402
from training import datasets  # noqa: E402
from training import thinker_train_common as C  # noqa: E402
from training.tokenizer import BPETokenizer, normalize  # noqa: E402

ORACLE = ROOT / "data" / "private" / "oracle"


# ------------------------------------------------------------------ data
def _tok(cfg) -> BPETokenizer:
    return BPETokenizer.load(ROOT / cfg["tokenizer"])


def encode_pair(tok, prompt: str, code: str, max_len: int | None = None):
    """-> (ids, out_mask) or None when it does not fit ``max_len`` / has unencodable text."""
    try:
        p = [BOS] + tok.encode(normalize(prompt)) + [SEP]
        c = tok.encode(normalize(code)) + [EOS]
    except ValueError:
        return None
    if max_len and len(p) + len(c) > max_len:
        return None
    return p + c, [False] * len(p) + [True] * len(c)


def _oracle_rows(run: str) -> dict[str, list[dict]]:
    out = {}
    for s in ("train", "val", "test"):
        f = ORACLE / run / f"{s}.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
        out[s] = [{"id": r.get("id"), "source": f"oracle:{r.get('source')}", "lang": r["lang"], "prompt": r["prompt"],
                   "code": r["code"], "tests": None, "origin": "oracle"} for r in rows if r.get("passed", True)]
    return out


def load_pairs(data: str = "auto", min_oracle: int = 200, seed: int = 0) -> dict:
    """``data``: ``auto`` (oracle run if it has >= min_oracle train pairs, else dataset records),
    ``oracle:<run>`` or ``dataset``.  Returns {"train","val","test","label"}; each pair carries ``origin``."""
    if data.startswith("oracle:") or data == "auto":
        runs = [data.split(":", 1)[1]] if data.startswith("oracle:") else \
            sorted(p.name for p in ORACLE.glob("*") if (p / "train.jsonl").exists()) if ORACLE.exists() else []
        best = max((_oracle_rows(r) | {"run": r} for r in runs), key=lambda d: len(d["train"]), default=None)
        if best is not None and (data.startswith("oracle:") or len(best["train"]) >= min_oracle):
            best["label"] = f"oracle:{best.pop('run')} (train={len(best['train'])})"
            return best
        n_or = len(best["train"]) if best else 0
        note = f"dataset-fallback (oracle train pairs {n_or} < {min_oracle})"
    else:
        note = "dataset"
    sp = datasets.split_records(datasets.load_all(), seed=seed)
    out = {s: [{**r, "origin": "dataset"} for r in rs] for s, rs in sp.items()}
    out["label"] = f"{note}; sources=" + json.dumps(datasets.counts(sum(sp.values(), [])), sort_keys=True)
    return out


def encode_split(tok, pairs, max_len):
    items, kept = [], []
    for r in pairs:
        e = encode_pair(tok, r["prompt"], r["code"], max_len)
        if e is not None:
            items.append(e); kept.append(r)
    return items, kept


# ------------------------------------------------------------------ evaluation
def make_evaluator(tok, cfg, val_pairs, run_gen: bool = True):
    t = cfg["train"]
    items, kept = encode_split(tok, val_pairs, t["max_len"])
    items, kept = items[:t["eval_pairs"]], kept[:t["eval_pairs"]]
    seq, mask = C.pad_batch(items) if items else (None, None)

    def evaluate(model) -> dict:
        model.eval()
        if seq is None:
            return {"n_val": 0}
        acc, nll = C.token_accuracy(model, seq, mask)
        out = {"token_acc": round(acc, 4), "nll": round(nll, 4), "n_val": len(items)}
        if run_gen and t["gen_pairs"] > 0:
            out.update(gen_metrics(model, tok, kept[:t["gen_pairs"]], t["gen_max_new"]))
        return out
    return evaluate


def gen_metrics(model, tok, pairs, max_new: int) -> dict:
    prompts = [[BOS] + tok.encode(normalize(r["prompt"])) for r in pairs]
    outs = C.generate_lists(model, prompts, max_new)
    parse = passed = exact = hit = tot = 0
    for r, o in zip(pairs, outs):
        gold = tok.encode(normalize(r["code"]))
        hit += sum(int(i < len(o) and o[i] == g) for i, g in enumerate(gold))   # position-aligned, student-only
        tot += len(gold)
        code = tok.decode(o)
        exact += int(code.strip() == normalize(r["code"]).strip())
        try:
            res = exec_judge.judge(r["lang"], code, r.get("tests") or "", timeout=10.0)
            parse += int(res["stage"] != "parse")
            passed += int(res["ok"])
        except Exception:
            pass
    n = max(len(pairs), 1)
    return {"gen_n": len(pairs), "gen_parse_rate": round(parse / n, 4), "gen_pass_rate": round(passed / n, 4),
            "gen_exact_rate": round(exact / n, 4),
            "gen_token_acc": round(hit / max(tot, 1), 4)}


# ------------------------------------------------------------------ core runner
def make_batcher(items, batch, seed, max_len):
    def get_batch(step):
        idx = C.sample_indices(len(items), batch, seed, step)
        return C.pad_batch([items[i] for i in idx], max_len)
    return get_batch


def cfg_fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def train_kind(cfg: dict, kind: str, seed: int, out_dir, data: dict, steps: int, resume=None, ctx=None,
               ram_gb=None, overfit: int = 0, log=print, tracker=None) -> dict:
    """Train one (kind, seed) under ``cfg``; nothing but kind/seed varies between control runs."""
    C.guard_ram(ram_gb)
    torch.set_num_threads(cfg.get("threads", 4))
    tok = _tok(cfg)
    t = cfg["train"]
    train_items, train_pairs = encode_split(tok, data["train"], t["max_len"])
    if overfit:
        train_items, train_pairs = train_items[:overfit], train_pairs[:overfit]
        val_pairs = train_pairs
    else:
        val_pairs = data["val"] or data["train"][-t["eval_pairs"]:]
    if not train_items:
        raise RuntimeError("no training pairs fit max_len")
    model, rep = C.build(kind, cfg, seed)
    trainer = C.Trainer(model, t, cfg["schedule"], out_dir, seed=seed, tag=f"{kind}/s{seed}", ctx=ctx, ram_gb=ram_gb)
    if tracker is not None:
        trainer.tracker = tracker
    evaluate = make_evaluator(tok, cfg, val_pairs)
    get_batch = make_batcher(train_items, min(t["batch"], len(train_items)) if overfit else t["batch"], seed, t["max_len"])
    t0 = time.time()
    res = trainer.run(get_batch, evaluate, steps, resume=resume, log=log)
    res.update({"kind": kind, "seed": seed, "params": rep, "n_train": len(train_items), "data": data["label"],
                "cfg_hash": cfg_fingerprint(cfg), "wall": round(time.time() - t0, 1)})
    Path(out_dir, "result.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def overfit_report(cfg, kind, n, steps, out_dir, data, seed=0, ram_gb=None, log=print, sched_override=None) -> dict:
    """Overfit gate: memorise ``n`` pairs; gate = teacher-forced acc >= 0.95 (greedy exact rate reported)."""
    cfg = json.loads(json.dumps(cfg))
    if sched_override:
        cfg["schedule"].update(sched_override)
    cfg["train"].update({"eval_pairs": n, "gen_pairs": min(n, cfg["train"]["gen_pairs"]), "eval_every": 0})
    res = train_kind(cfg, kind, seed, out_dir, data, steps, ram_gb=ram_gb, overfit=n, log=log)
    f = res["final"]
    res["overfit_gate"] = bool(f.get("token_acc", 0) >= 0.95)
    Path(out_dir, "result.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ------------------------------------------------------------------ controls
METRICS = ("token_acc", "gen_token_acc", "gen_parse_rate", "gen_pass_rate")


def run_controls(cfg: dict, kinds, seeds, out_dir, data, steps, ram_gb=None, log=print, tracker=None) -> dict:
    """Every kind x seed from ONE shared cfg (identical schedule, data, batches, budget).  Writes
    ``controls.json`` + ``controls.md``; CI = paired bootstrap of (real - control) over seeds."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp = cfg_fingerprint(cfg)
    rows = []
    for seed in seeds:
        for kind in kinds:
            r = train_kind(cfg, kind, seed, out / f"{kind}_s{seed}", data, steps, ram_gb=ram_gb, log=log,
                           tracker=tracker)
            assert r["cfg_hash"] == fp, "control runs must share one config"
            rows.append({"kind": kind, "seed": seed, "steps": r["steps"], "seconds": r["seconds"],
                         "trainable": r["params"]["trainable"], "cfg_hash": r["cfg_hash"], **r["final"]})
    table = summarize_controls(rows, kinds, seeds)
    res = {"cfg_hash": fp, "steps": steps, "seeds": list(seeds), "kinds": list(kinds), "data": data["label"],
           "rows": rows, "summary": table}
    (out / "controls.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "controls.md").write_text(controls_markdown(res))
    return res


def summarize_controls(rows, kinds, seeds) -> dict:
    by = {(r["kind"], r["seed"]): r for r in rows}
    summ = {}
    for k in kinds:
        d = {"trainable": by[(k, seeds[0])]["trainable"]}
        for m in METRICS:
            v = [by[(k, s)].get(m) for s in seeds if by[(k, s)].get(m) is not None]
            if v:
                d[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        if k != "real" and "real" in kinds:
            for m in METRICS:
                a = [by[("real", s)].get(m) for s in seeds]
                b = [by[(k, s)].get(m) for s in seeds]
                if None not in a and None not in b:
                    d[f"real_minus_{m}_ci95"] = C.bootstrap_ci(a, b)
        summ[k] = d
    return summ


def controls_markdown(res) -> str:
    L = [f"controls: cfg_hash={res['cfg_hash']} steps={res['steps']} seeds={res['seeds']} data={res['data']}", "",
         "| kind | trainable | token_acc | parse | pass | real-minus-kind token_acc CI95 |", "|---|---|---|---|---|---|"]
    for k, d in res["summary"].items():
        f = lambda m: f"{d[m]['mean']:.3f}+-{d[m]['std']:.3f}" if m in d else "-"
        ci = d.get("real_minus_token_acc_ci95")
        L.append(f"| {k} | {d['trainable']} | {f('token_acc')} | {f('gen_parse_rate')} | {f('gen_pass_rate')} | "
                 f"{'[%.3f, %.3f]' % tuple(ci) if ci else '-'} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ curriculum entry point
def run_stage(ctx, stage_cfg: dict, smoke: bool):
    """Stage 6 runner (``run(ctx, stage_cfg, smoke)`` protocol).  ``stage_cfg`` keys (all optional):
    kind, steps, controls (bool), seeds, kinds, data, thinker_config, overrides {dotted.key: value}."""
    from training.stages import StageResult
    ov = dict(stage_cfg.get("overrides", {}))
    if smoke:
        ov = {"model.k_steps": 2, "train.batch": 8, "train.max_len": 96, "train.eval_pairs": 8, "train.gen_pairs": 4,
              "train.gen_max_new": 48, "train.eval_every": 0, "train.ckpt_every": 0, "train.log_every": 5, "train.tf_lambda": 0.0, **ov}
    cfg = C.load_cfg(stage_cfg.get("thinker_config"), ov)
    steps = int(stage_cfg.get("steps", 30 if smoke else cfg["train"]["max_steps"]))
    if "wall_s" in ctx.budget:
        cfg["train"]["max_seconds"] = min(cfg["train"]["max_seconds"], ctx.budget["wall_s"])
    data = load_pairs(stage_cfg.get("data", cfg["distill"]["data"]), cfg["distill"]["min_oracle_pairs"], ctx.seed)
    out = Path(ctx.stage_dir)
    if stage_cfg.get("controls"):
        kinds = stage_cfg.get("kinds", ["gru"] if smoke else cfg["distill"]["controls_kinds"])
        seeds = stage_cfg.get("seeds", [0, 1] if smoke else cfg["distill"]["controls_seeds"])
        res = run_controls(cfg, kinds, seeds, out, data, steps, ram_gb=ctx.ram_gb,
                           tracker=getattr(ctx, "tracker", None))
        s = res["summary"]
        beat = all(s[k].get("real_minus_token_acc_ci95", [-1])[0] > 0
                   for k in ("shuffled", "random_sparse") if k in s) if "real" in kinds else False
        return StageResult(metrics={"controls": s, "cfg_hash": res["cfg_hash"], "data": data["label"]},
                           gate_passed=bool(beat), artifacts=[str(out / "controls.md")])
    kind = stage_cfg.get("kind", "real")
    res = train_kind(cfg, kind, ctx.seed, out, data, steps, resume=stage_cfg.get("resume"), ctx=ctx, ram_gb=ctx.ram_gb)
    f = res["final"]
    floor = stage_cfg.get("min_token_acc", 0.2 if smoke else 0.5)
    return StageResult(metrics={**f, "steps": res["steps"], "kind": kind, "data": data["label"]},
                       gate_passed=bool(f.get("token_acc", 0) >= floor), artifacts=[str(out / "ckpt.pt")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="real", choices=KINDS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/distill/run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=None)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--overfit", type=int, default=0, help="memorise this many pairs (overfit gate)")
    ap.add_argument("--data", default=None, help="auto | dataset | oracle:<run>")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ram-gb", type=float, default=None)
    ap.add_argument("--k-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--tbptt", type=int, default=None)
    ap.add_argument("--circuit", default=None)
    ap.add_argument("--a-frac", type=float, default=None)
    ap.add_argument("--b-frac", type=float, default=None)
    ap.add_argument("--schedule", choices=["gated", "fixed"], default=None)
    ap.add_argument("--mix-mode", choices=["sequential", "parallel"], default=None)
    ap.add_argument("--tf-lambda", type=float, default=None, help="teacher-forced loss weight in phases B/C")
    ap.add_argument("--mix-temperature", type=float, default=None, help="0 = argmax replacement")
    ap.add_argument("--a-thresh", type=float, default=None)
    ap.add_argument("--b-floor", type=float, default=None)
    ap.add_argument("--gate-window", type=int, default=None)
    ap.add_argument("--gate-min-steps", type=int, default=None)
    ap.add_argument("--eval-pairs", type=int, default=None)
    ap.add_argument("--gen-pairs", type=int, default=None)
    ap.add_argument("--gen-max-new", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--min-oracle-pairs", type=int, default=None)
    a = ap.parse_args(argv)
    ov = {k: v for k, v in {
        "model.k_steps": a.k_steps, "model.tbptt": a.tbptt, "model.circuit": a.circuit, "train.batch": a.batch,
        "train.max_len": a.max_len, "train.max_seconds": a.max_seconds, "schedule.a_frac": a.a_frac,
        "schedule.b_frac": a.b_frac,
        "schedule.mode": a.schedule, "train.mix_mode": a.mix_mode, "train.tf_lambda": a.tf_lambda,
        "train.mix_temperature": a.mix_temperature, "schedule.a_thresh": a.a_thresh, "schedule.b_floor": a.b_floor,
        "schedule.window": a.gate_window, "schedule.min_steps": a.gate_min_steps, "train.eval_pairs": a.eval_pairs, "train.gen_pairs": a.gen_pairs,
        "train.gen_max_new": a.gen_max_new, "train.eval_every": a.eval_every, "train.log_every": a.log_every,
        "distill.min_oracle_pairs": a.min_oracle_pairs}.items() if v is not None}
    cfg = C.load_cfg(a.config, ov)
    steps = a.steps or cfg["train"]["max_steps"]
    C.guard_ram(a.ram_gb)
    data = load_pairs(a.data or cfg["distill"]["data"], cfg["distill"]["min_oracle_pairs"], a.seed)
    print("data:", data["label"], {s: len(data[s]) for s in ("train", "val", "test")}, flush=True)
    if a.controls:
        res = run_controls(cfg, a.kinds or cfg["distill"]["controls_kinds"], a.seeds or cfg["distill"]["controls_seeds"],
                           a.out, data, steps, ram_gb=a.ram_gb)
        print(controls_markdown(res))
    elif a.overfit:
        # memorisation is a teacher-forcing property: default to schedule A only (pass --a-frac/--b-frac for A->B->C)
        so = None if (a.a_frac is not None or a.b_frac is not None or a.schedule) else {"a_frac": 1.0, "b_frac": 0.0}
        res = overfit_report(cfg, a.kind, a.overfit, steps, a.out, data, a.seed, a.ram_gb, sched_override=so)
        print(json.dumps({k: res[k] for k in ("kind", "steps", "seconds", "final", "overfit_gate")}, indent=1))
    else:
        res = train_kind(cfg, a.kind, a.seed, a.out, data, steps, resume=a.resume, ram_gb=a.ram_gb)
        print(json.dumps({k: res[k] for k in ("kind", "steps", "stopped", "seconds", "final")}, indent=1))
    from backend import memory_budget
    print(f"peak RSS {memory_budget.peak_rss() / memory_budget.GB:.2f} GB")


if __name__ == "__main__":
    main()
