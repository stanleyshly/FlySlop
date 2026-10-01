"""Edit-token supervised trainer for the stage 3-4 thinker (P6b).

Sequence: ``BOS target SEP buffer-with-markers SEP actions EOS``; loss on actions + EOS only.  Markers are the
reserved ids CUR=17 (cursor / selection head), SEL_A=18 / SEL_B=19 (selection start / end).  ``actions`` are
``backend.keyplan.oracle_edits`` tokens: control ids 0-15 (arrows, BS, DEL, ...) and BPE ids for typed text.

* stage 3 examples: dataset code snippets with random edits (delete / insert / replace, random cursor,
  optional selection).  stage 4: ``error_inject.corrupt_buffer`` corruptions (wrong / missing / duplicated
  characters, bad indents, stray cursor) mixed with 30% random edits.
* metric (stage 3 gate): predicted tokens are applied through ``Editor``; success = target reached within
  ``ratio`` (1.5) x the oracle's keystrokes.  ``replan``: predict -> apply -> observe -> predict again, up to
  N rounds, cost summed over rounds.

Full runs (offline):
    uv run --extra training python -m training.edit_trainer --stage 3 --kind real --steps 20000 --out runs/edit/s3_real
    uv run --extra training python -m training.edit_trainer --stage 4 --kind real --steps 20000 --out runs/edit/s4_real
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import keyplan  # noqa: E402
from backend.connectome.thinker import BOS, EOS, KINDS, SEP  # noqa: E402
from backend.editor import Editor  # noqa: E402
from backend.error_inject import corrupt_buffer  # noqa: E402
from training import thinker_train_common as C  # noqa: E402
from training.distill import _tok, load_pairs  # noqa: E402
from training.tokenizer import SPECIALS, normalize  # noqa: E402

CUR, SEL_A, SEL_B = 17, 18, 19
_PRINTABLE = "".join(chr(c) for c in range(33, 127))


# ------------------------------------------------------------------ encoding
def enc_state(tok, target: str, buffer: str, cursor: int, anchor: int | None = None) -> list[int]:
    """``BOS target SEP buffer(with markers)``.  A generation prompt (``model.generate`` appends the SEP)."""
    anchor = cursor if anchor is None else anchor
    marks = [(cursor, 1, CUR)]
    if anchor != cursor:
        lo, hi = sorted((anchor, cursor))
        marks += [(lo, 2, SEL_A), (hi, 0, SEL_B)]
    ids, prev = [BOS] + tok.encode(target) + [SEP], 0
    for pos, _, m in sorted(marks):
        ids += tok.encode(buffer[prev:pos]) + [m]
        prev = pos
    return ids + tok.encode(buffer[prev:])


def enc_actions(tok, toks: list[str]) -> list[int]:
    ids: list[int] = []
    for t in toks:
        ids += [SPECIALS.index(t)] if t in SPECIALS else tok.encode(t)
    return ids


def cost_of(tok, ids) -> int:
    return keyplan.keystroke_cost([i for i in ids if i not in (EOS, 0, 1)], tok)


# ------------------------------------------------------------------ example generation
def _snippet(code: str, rng: random.Random, max_chars: int) -> str | None:
    lines = normalize(code).split("\n")
    for _ in range(6):
        i = rng.randrange(len(lines))
        s = "\n".join(lines[i:i + rng.randint(1, 3)])
        if 3 <= len(s) <= max_chars:
            return s
    s = lines[rng.randrange(len(lines))]
    return s if 3 <= len(s) <= max_chars else None


def random_edit(target: str, rng: random.Random) -> tuple[str, int, int]:
    """(buffer, cursor, anchor): the target with 1-3 random deletions / insertions / replacements, a random
    cursor and (30%) a random selection."""
    buf = target
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(len(buf) + 1)
        op = rng.choice(("delete", "insert", "replace"))
        n = rng.randint(1, min(6, max(1, len(buf) - i)))
        junk = "".join(rng.choice(_PRINTABLE + "  ") for _ in range(rng.randint(1, 5)))
        if op == "delete" and i < len(buf):
            buf = buf[:i] + buf[i + n:]
        elif op == "replace" and i < len(buf):
            buf = buf[:i] + junk + buf[i + n:]
        else:
            buf = buf[:i] + junk + buf[i:]
    cursor = rng.randint(0, len(buf))
    anchor = rng.randint(0, len(buf)) if rng.random() < 0.3 else cursor
    return buf, cursor, anchor


def make_example(tok, code: str, seed: int, stage: int, ecfg: dict, max_len: int):
    """One supervised example dict or None (unencodable / too long / oracle failure)."""
    rng = random.Random(seed)
    target = _snippet(code, rng, ecfg["max_target_chars"])
    if target is None:
        return None
    if rng.random() < 0.05:                                    # already done: model must emit just EOS
        buf, cursor, anchor = target, rng.randint(0, len(target)), None
        anchor = cursor
    elif stage >= 4 and rng.random() < 0.7:
        cb = corrupt_buffer(target, rng.randint(0, len(target)), seed=rng.randrange(1 << 30),
                            n=rng.choice(ecfg["corrupt_n"]))
        buf, cursor, anchor = cb.text, cb.cursor, cb.cursor
    else:
        buf, cursor, anchor = random_edit(target, rng)
    sel = {"anchor": anchor, "head": cursor} if anchor != cursor else None
    try:
        toks, cost = keyplan.oracle_plan(buf, cursor, sel, target)
        state = enc_state(tok, target, buf, cursor, anchor)
        act = enc_actions(tok, toks)
    except (ValueError, TypeError):
        return None
    if len(state) + 1 + len(act) + 1 > max_len or len(act) + 1 > ecfg["max_new"]:
        return None
    if cost_of(tok, act) != cost:                              # supervision must round-trip through the keys
        return None
    return {"target": target, "buffer": buf, "cursor": cursor, "anchor": anchor, "state": state, "act": act,
            "cost": cost, "ids": state + [SEP] + act + [EOS],
            "mask": [False] * (len(state) + 1) + [True] * (len(act) + 1)}


def build_examples(tok, records, n: int, seed: int, stage: int, ecfg: dict, max_len: int) -> list[dict]:
    out, i = [], 0
    while len(out) < n and i < n * 20:
        r = records[(seed * 7919 + i) % len(records)]
        ex = make_example(tok, r["code"], seed * 1_000_003 + i, stage, ecfg, max_len)
        i += 1
        if ex:
            out.append(ex)
    return out


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def _predict(model, tok, states: list[list[int]], max_new: int) -> list[list[int]]:
    return C.generate_lists(model, states, max_new)


def _apply(tok, ed: Editor, ids: list[int]) -> bool:
    try:
        keyplan.apply_tokens(ed, ids, tok)
        return True
    except Exception:
        return False


@torch.no_grad()
def edit_metrics(model, tok, exs: list[dict], ecfg: dict) -> dict:
    """First-round gate metric and the re-plan loop (batched over examples)."""
    ratio, rounds, max_new = ecfg["ratio"], ecfg["replan_rounds"], ecfg["max_new"]
    model.eval()
    eds = [Editor(e["buffer"], e["cursor"], e["anchor"]) for e in exs]
    spent = [0] * len(exs)
    first_ok = [False] * len(exs)
    reached = [False] * len(exs)
    exact_act = 0
    for rd in range(rounds):
        live = [i for i, e in enumerate(exs) if not reached[i]]
        if not live:
            break
        preds = _predict(model, tok, [enc_state(tok, exs[i]["target"], eds[i].text, eds[i].cursor, eds[i].anchor)
                                      for i in live], max_new)
        for i, p in zip(live, preds):
            if rd == 0:
                exact_act += int(p == exs[i]["act"])
            _apply(tok, eds[i], p)
            spent[i] += cost_of(tok, p)
            reached[i] = eds[i].text == exs[i]["target"]
            if rd == 0:
                first_ok[i] = reached[i] and spent[i] <= ratio * exs[i]["cost"]
    n = max(len(exs), 1)
    within = [reached[i] and spent[i] <= ratio * max(exs[i]["cost"], 1) for i in range(len(exs))]
    over = [spent[i] / max(exs[i]["cost"], 1) for i in range(len(exs)) if reached[i] and exs[i]["cost"] > 0]
    return {"first_round_ok": round(sum(first_ok) / n, 4), "replan_reach": round(sum(reached) / n, 4),
            "replan_within_ratio": round(sum(within) / n, 4), "act_exact_first": round(exact_act / n, 4),
            "mean_cost_ratio": round(sum(over) / max(len(over), 1), 3), "n_eval": len(exs), "rounds": rounds}


def make_evaluator(tok, cfg, val_exs):
    ecfg, t = cfg["edit"], cfg["train"]
    items = [(e["ids"], e["mask"]) for e in val_exs]
    seq, mask = C.pad_batch(items)

    def evaluate(model) -> dict:
        model.eval()
        acc, nll = C.token_accuracy(model, seq, mask)
        return {"token_acc": round(acc, 4), "nll": round(nll, 4), **edit_metrics(model, tok, val_exs, ecfg)}
    return evaluate


# ------------------------------------------------------------------ runner
def train_edit(cfg, kind, seed, out_dir, steps, stage=3, resume=None, ctx=None, ram_gb=None, log=print) -> dict:
    C.guard_ram(ram_gb)
    torch.set_num_threads(cfg.get("threads", 4))
    tok, t, ecfg = _tok(cfg), cfg["train"], cfg["edit"]
    data = load_pairs("dataset", seed=seed)
    t0 = time.time()
    train_exs = build_examples(tok, data["train"], ecfg["train_examples"], seed, stage, ecfg, t["max_len"])
    val_exs = build_examples(tok, data["val"] or data["train"], ecfg["val_examples"], seed + 10_000, stage, ecfg, t["max_len"])
    log(f"[edit s{stage}] {len(train_exs)} train / {len(val_exs)} val examples in {time.time() - t0:.1f}s; "
        f"oracle cost mean {sum(e['cost'] for e in train_exs) / max(len(train_exs), 1):.1f}")
    val_exs = val_exs[:t["eval_pairs"]]
    items = [(e["ids"], e["mask"]) for e in train_exs]

    def get_batch(step):
        idx = C.sample_indices(len(items), t["batch"], seed, step)
        return C.pad_batch([items[i] for i in idx], t["max_len"])

    model, rep = C.build(kind, cfg, seed)
    trainer = C.Trainer(model, t, ecfg.get("schedule", cfg["schedule"]), out_dir, seed=seed,
                        tag=f"edit{stage}/{kind}", ctx=ctx, ram_gb=ram_gb)
    res = trainer.run(get_batch, make_evaluator(tok, cfg, val_exs), steps, resume=resume, log=log)
    res.update({"kind": kind, "seed": seed, "stage": stage, "params": rep, "n_train": len(train_exs),
                "cfg_hash": __import__("training.distill", fromlist=["x"]).cfg_fingerprint(cfg)})
    Path(out_dir, "result.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def run_stage(ctx, stage_cfg: dict, smoke: bool, stage: int = 3):
    """Stage 3 / 4 thinker runner (``run(ctx, stage_cfg, smoke)`` protocol, ``stage`` selects the example mix).
    Gate: stage 3 first_round_ok >= 0.95; stage 4 replan_reach >= 0.90."""
    from training.stages import StageResult
    ov = dict(stage_cfg.get("overrides", {}))
    if smoke:
        ov = {"model.k_steps": 2, "train.batch": 8, "train.max_len": 128, "train.eval_pairs": 8, "train.eval_every": 0,
              "train.ckpt_every": 0, "train.log_every": 5, "edit.train_examples": 64, "edit.val_examples": 8, **ov}
    cfg = C.load_cfg(stage_cfg.get("thinker_config"), ov)
    steps = int(stage_cfg.get("steps", 20 if smoke else cfg["train"]["max_steps"]))
    if "wall_s" in ctx.budget:
        cfg["train"]["max_seconds"] = min(cfg["train"]["max_seconds"], ctx.budget["wall_s"])
    res = train_edit(cfg, stage_cfg.get("kind", "real"), ctx.seed, ctx.stage_dir, steps, stage=stage,
                     resume=stage_cfg.get("resume"), ctx=ctx, ram_gb=ctx.ram_gb)
    f = res["final"]
    metric, floor = ("first_round_ok", 0.95) if stage == 3 else ("replan_reach", 0.90)
    floor = stage_cfg.get("floor", 0.0 if smoke else floor)
    return StageResult(metrics={**f, "steps": res["steps"], "stage": stage}, gate_passed=bool(f[metric] >= floor),
                       artifacts=[str(Path(ctx.stage_dir) / "ckpt.pt")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(3, 4), default=3)
    ap.add_argument("--kind", default="real", choices=KINDS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/edit/run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ram-gb", type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--k-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--tbptt", type=int, default=None)
    ap.add_argument("--train-examples", type=int, default=None)
    ap.add_argument("--val-examples", type=int, default=None)
    ap.add_argument("--eval-pairs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    a = ap.parse_args(argv)
    ov = {k: v for k, v in {"model.k_steps": a.k_steps, "model.tbptt": a.tbptt, "train.batch": a.batch,
                            "train.max_len": a.max_len, "train.max_seconds": a.max_seconds,
                            "edit.train_examples": a.train_examples, "edit.val_examples": a.val_examples,
                            "train.eval_pairs": a.eval_pairs, "train.eval_every": a.eval_every,
                            "train.log_every": a.log_every}.items() if v is not None}
    cfg = C.load_cfg(a.config, ov)
    res = train_edit(cfg, a.kind, a.seed, a.out, a.steps or cfg["train"]["max_steps"], a.stage, a.resume,
                     ram_gb=a.ram_gb)
    print(json.dumps({k: res[k] for k in ("kind", "stage", "steps", "stopped", "seconds", "final")}, indent=1))
    from backend import memory_budget
    print(f"peak RSS {memory_budget.peak_rss() / memory_budget.GB:.2f} GB")


if __name__ == "__main__":
    main()
