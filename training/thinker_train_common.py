"""Shared machinery for the thinker trainers (P6b): distillation (stage 6) and the edit-token trainer
(stages 3-4).

* ``load_cfg`` reads ``training/thinker.json``.
* Batches are left-padded ``(seq [B,T], out_mask [B,T])``; ``out_mask[b,t]`` marks tokens the model must
  predict (loss) and that scheduled sampling may replace by the student's own sample.
* ``Trainer.run`` is the one training loop: truncated BPTT (backward per ``tbptt`` chunk), the A->B->C
  scheduled-sampling schedule logged per step, wall/step budgets, checkpoint + resume, RAM guard.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend import memory_budget
from backend.connectome.thinker import PAD, EOS, SEP, ThinkerConfig, build_model, param_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training" / "thinker.json"
_WATCHDOG = []


def load_cfg(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = json.loads(Path(path or CONFIG).read_text())
    for dotted, v in (overrides or {}).items():
        d = cfg
        *head, last = dotted.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = v
    return cfg


def model_config(cfg: dict, seed: int = 0, **over) -> ThinkerConfig:
    names = {f.name for f in fields(ThinkerConfig)}
    kw = {k: v for k, v in cfg["model"].items() if k in names}
    kw.update(over)
    return ThinkerConfig(seed=seed, **kw)


def guard_ram(ram_gb: float | None):
    """Start the RSS watchdog once (aborts the process above the cap)."""
    if not _WATCHDOG:
        _WATCHDOG.append(memory_budget.start_watchdog(config_gb=ram_gb))


def estimate_bytes(model, batch: int, chunk: int) -> int:
    """Rough peak: params x (weights, grads, Adam m and v) plus BPTT activations for one chunk."""
    p = sum(x.numel() for x in model.parameters()) * 4 * 4
    core = getattr(model, "core", None)
    if core is not None:
        act = batch * core.n * 4 * (core.k_steps * 3 + 2) * chunk
        act += core.n_edges * 4 * 8
    else:
        act = batch * getattr(model, "hidden", 256) * 4 * 24 * chunk
    act += batch * chunk * model.cfg.vocab_size * 4 * 3
    return int(p + act + 200 * (1 << 20))


# ------------------------------------------------------------------ schedule
def schedule_p(step: int, total: int, sched: dict) -> tuple[str, float]:
    """Scheduled-sampling phase and replacement probability for 0-based ``step`` of ``total``.
    A: teacher forcing (p=0).  B: p steps through ``b_levels`` (0.25, 0.5, 0.75, 0.95) in equal blocks.
    C: student-only (p=1: every output-region input is the student's own free-running sample)."""
    a_end = sched["a_frac"] * total
    b_end = a_end + sched["b_frac"] * total
    if step < a_end:
        return "A", 0.0
    if step < b_end:
        lv = sched["b_levels"]
        i = min(int((step - a_end) / max(b_end - a_end, 1e-9) * len(lv)), len(lv) - 1)
        return "B", float(lv[i])
    return "C", 1.0


GATE_DEFAULTS = {"mode": "gated", "window": 20, "a_thresh": 0.9, "b_floor": 0.85, "min_steps": 20}


class FixedSchedule:
    """Step-fraction A->B->C (``schedule.mode = "fixed"``); transitions are logged with reason ``fixed``."""

    def __init__(self, sched: dict, total: int):
        self.sched, self.total, self.last = sched, total, None

    def current(self, step: int) -> tuple[str, float]:
        return schedule_p(step, self.total, self.sched)

    def observe(self, step: int, acc: float) -> list[dict]:
        cur = self.current(step)          # ``step`` = number of completed steps = index of the next one
        ev = []
        if self.last is not None and cur != self.last:
            ev.append({"event": "transition", "step": step, "from": self.last[0], "to": cur[0], "p": cur[1],
                       "from_p": self.last[1], "reason": "fixed"})
        self.last = cur
        return ev

    def dump(self) -> dict:
        return {"mode": "fixed", "last": list(self.last) if self.last else None}

    def load(self, d: dict | None):
        if d and d.get("mode") == "fixed" and d.get("last"):
            self.last = tuple(d["last"])


class GatedSchedule:
    """Competence-gated A->B->C.  Stay in A until the rolling training token accuracy (``window`` steps)
    reaches ``a_thresh``; then advance through ``b_levels`` one at a time, each only once the rolling
    accuracy at the current p is >= ``b_floor``; then C (p=1).  Every stage also has a step cap (from
    ``a_frac``/``b_frac`` of the run, so gating can only advance EARLIER than the fixed schedule) after which
    it advances anyway (reason ``step_cap``).  ``a_frac >= 1`` pins teacher forcing (A only)."""

    def __init__(self, sched: dict, total: int):
        g = {**GATE_DEFAULTS, **{k: v for k, v in sched.items() if k in GATE_DEFAULTS}}
        self.g = g
        self.levels = list(sched.get("b_levels", []))
        b_steps = sched.get("b_frac", 0.0) * total
        self.a_cap = max(1, int(sched.get("a_frac", 1.0) * total))
        self.lv_cap = max(1, int(b_steps / len(self.levels))) if self.levels and b_steps > 0 else 0
        self.pinned = sched.get("a_frac", 1.0) >= 1.0
        self.phase, self.idx, self.since, self.accs = "A", 0, 0, []

    def current(self, step: int) -> tuple[str, float]:
        if self.phase == "A":
            return "A", 0.0
        if self.phase == "B":
            return "B", float(self.levels[self.idx])
        return "C", 1.0

    def observe(self, step: int, acc: float) -> list[dict]:
        if self.phase == "C" or self.pinned:
            return []
        g = self.g
        self.accs = (self.accs + [float(acc)])[-g["window"]:]
        self.since += 1
        roll = sum(self.accs) / len(self.accs)
        thr = g["a_thresh"] if self.phase == "A" else g["b_floor"]
        cap = self.a_cap if self.phase == "A" else self.lv_cap
        if len(self.accs) >= g["window"] and self.since >= g["min_steps"] and roll >= thr:
            reason = "threshold"
        elif self.since >= cap:
            reason = "step_cap"
        else:
            return []
        frm, from_p = self.current(step)
        if self.phase == "A":
            self.phase, self.idx = ("B", 0) if self.levels and self.lv_cap else ("C", 0)
        elif self.idx + 1 < len(self.levels):
            self.idx += 1
        else:
            self.phase = "C"
        self.since, self.accs = 0, []
        to, p = self.current(step)
        return [{"event": "transition", "step": step, "from": frm, "to": to, "p": p, "from_p": from_p,
                 "reason": reason, "rolling_acc": round(roll, 4), "threshold": thr}]

    def dump(self) -> dict:
        return {"mode": "gated", "phase": self.phase, "idx": self.idx, "since": self.since, "accs": self.accs}

    def load(self, d: dict | None):
        if d and d.get("mode") == "gated":
            self.phase, self.idx, self.since, self.accs = d["phase"], d["idx"], d["since"], list(d["accs"])


def make_schedule(sched: dict, total: int):
    return (FixedSchedule if sched.get("mode", "gated") == "fixed" else GatedSchedule)(sched, total)


# ------------------------------------------------------------------ batching
def pad_batch(items, max_len: int | None = None):
    """items: [(ids, out_mask)] -> left-padded (seq [B,T] long, mask [B,T] bool)."""
    T = max(len(i) for i, _ in items)
    if max_len:
        T = min(T, max_len)
    seq = torch.full((len(items), T), PAD, dtype=torch.long)
    msk = torch.zeros((len(items), T), dtype=torch.bool)
    for b, (ids, m) in enumerate(items):
        ids, m = ids[-T:], m[-T:]
        seq[b, T - len(ids):] = torch.tensor(ids)
        msk[b, T - len(ids):] = torch.tensor(m, dtype=torch.bool)
    return seq, msk


def step_rng(seed: int, step: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, step, salt])


def sample_indices(n: int, batch: int, seed: int, step: int) -> list[int]:
    """Batch indices depend only on (seed, step): identical across model kinds and after a resume."""
    return step_rng(seed, step).choice(n, size=min(batch, n), replace=n < batch).tolist()


# ------------------------------------------------------------------ training pieces
@torch.no_grad()
def mix_inputs(model, x, mask, p: float, gen: torch.Generator, temperature: float = 1.0):
    """Sequential scheduled sampling.  Free-runs the student over ``x`` [B,T]; at each output-region
    position (``mask``) the input is replaced by the student's sample from the previous step's logits with
    probability ``p``.  p=1 is student-only decoding.  Returns (mixed x, replaced fraction)."""
    B, T = x.shape
    out = x.clone()
    state = model.init_state(B, x.device)
    logits, n_rep, n_all = None, 0, 0
    with model._weights():
        for t in range(T):
            inp = out[:, t]
            if t > 0 and mask[:, t].any():
                lg = logits.clone()
                lg[:, PAD] = -float("inf")
                if temperature and temperature > 0:
                    pred = torch.multinomial((lg / temperature).softmax(-1), 1, generator=gen)[:, 0]
                else:
                    pred = lg.argmax(-1)
                rep = mask[:, t] & (torch.rand(B, generator=gen) < p)
                inp = torch.where(rep, pred, inp)
                out[:, t] = inp
                n_rep += int(rep.sum())
                n_all += int(mask[:, t].sum())
            logits, state = model.step(inp, state)
    return out, n_rep / max(n_all, 1)


def _pass(model, x, y, m, chunk: int, n: int, weight: float, grad: bool, gen=None, temperature=None):
    """One chunked forward over inputs ``x`` (backward per chunk when ``grad``).  With ``temperature`` set
    also returns the student's per-position next-token choice (argmax when 0/None-like, else a sample)."""
    preds = torch.zeros_like(x) if gen is not None else None
    tot, hit = 0.0, 0
    with torch.set_grad_enabled(grad):
        for t0, logits in model.iter_chunks(x, chunk=chunk):
            c = logits.shape[1]
            yc, mc = y[:, t0:t0 + c], m[:, t0:t0 + c]
            if preds is not None:
                lg = logits.detach().clone()
                lg[..., PAD] = -float("inf")
                if temperature and temperature > 0:
                    B, _, V = lg.shape
                    pr = torch.multinomial((lg / temperature).softmax(-1).reshape(-1, V), 1, generator=gen).view(B, c)
                else:
                    pr = lg.argmax(-1)
                preds[:, t0:t0 + c] = pr
            if not mc.any():
                continue
            loss = nn.functional.cross_entropy(logits[mc], yc[mc], reduction="sum") / n
            if grad:
                (loss * weight).backward()
            tot += float(loss.detach())
            hit += int((logits.detach().argmax(-1)[mc] == yc[mc]).sum())
    return tot, hit / n, preds


def mix_parallel(x, m, preds, p: float, gen: torch.Generator):
    """One-pass (Bengio et al. 2015) scheduled sampling: input t+1 is replaced with probability ``p`` by the
    student's choice from the teacher-forced pass at t (conditioned on the GOLD prefix)."""
    out = x.clone()
    rep = m[:, :-1] & (torch.rand(m[:, :-1].shape, generator=gen) < p)
    out[:, 1:] = torch.where(rep, preds[:, :-1], x[:, 1:])
    return out, float(rep.sum()) / max(int(m[:, :-1].sum()), 1)


def train_step(model, opt, seq, mask, p: float, gen, chunk: int, temperature: float, clip: float,
               mode: str = "sequential", tf_lambda: float = 0.0):
    """One optimiser step on a batch; backward per TBPTT chunk.  ``mode``: ``sequential`` (no-grad free-run
    mixing, ``mix_inputs``) or ``parallel`` (per-token Bernoulli replacement from the teacher-forced pass).
    With p>0 and ``tf_lambda``>0 a teacher-forced loss (weight lambda) is added to the mixed-input loss.
    Returns (mixed-pass loss, mixed-pass acc, mixed fraction, teacher-forced acc or None)."""
    x, y, m = seq[:, :-1], seq[:, 1:], mask[:, 1:]
    n = max(int(m.sum()), 1)
    opt.zero_grad(set_to_none=True)
    frac, tf_acc = 0.0, None
    if p <= 0:
        loss, acc, _ = _pass(model, x, y, m, chunk, n, 1.0, True)
        tf_acc = acc
    else:
        preds = None
        if mode == "parallel":       # the TF pass doubles as the sampler (and carries the lambda loss)
            tf_loss, tf_acc, preds = _pass(model, x, y, m, chunk, n, tf_lambda, tf_lambda > 0, gen, temperature)
            xm, frac = mix_parallel(x, m, preds, p, gen)
        else:
            xm, frac = mix_inputs(model, x, mask[:, :-1], p, gen, temperature)
            if tf_lambda > 0:
                _, tf_acc, _ = _pass(model, x, y, m, chunk, n, tf_lambda, True)
        loss, acc, _ = _pass(model, xm, y, m, chunk, n, 1.0, True)
    if clip:
        nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], clip)
    opt.step()
    return loss, acc, frac, tf_acc


@torch.no_grad()
def token_accuracy(model, seq, mask, chunk: int = 64, batch: int = 32) -> tuple[float, float]:
    """Teacher-forced (accuracy, mean NLL) over the output-region tokens."""
    hit = nll = n = 0
    for b0 in range(0, seq.shape[0], batch):
        s, mk = seq[b0:b0 + batch], mask[b0:b0 + batch]
        x, y, m = s[:, :-1], s[:, 1:], mk[:, 1:]
        for t0, logits in model.iter_chunks(x, chunk=chunk):
            c = logits.shape[1]
            yc, mc = y[:, t0:t0 + c], m[:, t0:t0 + c]
            if mc.any():
                hit += int((logits.argmax(-1)[mc] == yc[mc]).sum())
                nll += float(nn.functional.cross_entropy(logits[mc], yc[mc], reduction="sum"))
                n += int(mc.sum())
    return hit / max(n, 1), nll / max(n, 1)


@torch.no_grad()
def generate_lists(model, prompts: list[list[int]], max_new: int, batch: int = 16) -> list[list[int]]:
    """Greedy continuations of ``prompt SEP`` (SEP added by ``model.generate``), cut at EOS."""
    outs = []
    for b0 in range(0, len(prompts), batch):
        g = model.generate(prompts[b0:b0 + batch], max_new=max_new)
        for row in g.tolist():
            row = row[:row.index(EOS)] if EOS in row else [t for t in row if t != PAD]
            outs.append(row)
    return outs


def make_optimizer(model, lr: float, conn_lr: float):
    core = [p for n, p in model.named_parameters() if n.startswith(("core.", "cell.")) and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("core.", "cell.")) and p.requires_grad]
    groups = [{"params": rest, "lr": lr, "base": lr}]
    if core:
        groups.append({"params": core, "lr": conn_lr, "base": conn_lr})
    return torch.optim.Adam(groups)


# ------------------------------------------------------------------ trainer
class Trainer:
    """One loop for every thinker trainer.  ``get_batch(step) -> (seq, mask)`` must depend only on
    (seed, step); ``evaluate(model) -> dict`` is called every ``eval_every`` steps and at the end."""

    def __init__(self, model, tcfg: dict, sched: dict, out_dir, seed: int = 0, tag: str = "", ctx=None,
                 ram_gb: float | None = None, tracker=None):
        self.model, self.t, self.sched, self.seed, self.tag, self.ctx = model, tcfg, sched, seed, tag, ctx
        self.tracker = tracker or getattr(ctx, "tracker", None)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.opt = make_optimizer(model, tcfg["lr"], tcfg["conn_lr"])
        self.step, self.elapsed, self.history, self.log_path = 0, 0.0, [], self.out / "schedule.jsonl"
        self.ram_gb = ram_gb
        self.sch, self.resume_extra = None, None

    # -- checkpoint
    def save(self, name="ckpt.pt", extra=None):
        path = self.out / name
        tmp = path.with_suffix(".tmp")
        extra = {"sched": self.sch.dump(), **(extra or {})} if getattr(self, "sch", None) else (extra or {})
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
                    "elapsed": self.elapsed, "history": self.history, "seed": self.seed, "kind": self.model.kind,
                    "cfg": self.model.cfg.__dict__, "extra": extra}, tmp)
        tmp.replace(path)
        return path

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["kind"] != self.model.kind:
            raise ValueError(f"checkpoint kind {ck['kind']} != model kind {self.model.kind}")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.step, self.elapsed, self.history = ck["step"], ck["elapsed"], ck["history"]
        self.resume_extra = ck.get("extra")
        return ck

    def run(self, get_batch, evaluate, total_steps: int, resume=None, log=print):
        t, model = self.t, self.model
        if resume:
            self.load(resume)
            log(f"[{self.tag}] resumed at step {self.step}")
        elif self.log_path.exists():
            self.log_path.unlink()
        chunk = t.get("chunk") or model.cfg.tbptt or 32
        seq0, _ = get_batch(self.step)
        memory_budget.check_fits(estimate_bytes(model, seq0.shape[0], chunk), f"{self.tag} training",
                                 raise_error=True, config_gb=self.ram_gb)
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + self.step)
        self.sch = make_schedule(self.sched, total_steps)
        if resume:
            self.sch.load((self.resume_extra or {}).get("sched"))
        stopped, t0, base = "steps", time.time(), self.elapsed
        last_eval = None
        try:
            while self.step < total_steps:
                self.elapsed = base + time.time() - t0
                if self.elapsed > t["max_seconds"]:
                    stopped = "wall"
                    break
                phase, p = self.sch.current(self.step)
                warm = min(1.0, (self.step + 1) / max(t.get("warmup", 1), 1))
                for g in self.opt.param_groups:
                    g["lr"] = g["base"] * warm
                seq, mask = get_batch(self.step)
                gen.manual_seed(self.seed * 1_000_003 + self.step)
                model.train()
                s0 = time.time()
                loss, acc, frac, tf_acc = train_step(
                    model, self.opt, seq, mask, p, gen, chunk, t.get("mix_temperature", 1.0), t.get("clip", 1.0),
                    mode=t.get("mix_mode", "sequential"), tf_lambda=t.get("tf_lambda", 0.0) if p > 0 else 0.0)
                self.step += 1
                row = {"step": self.step, "phase": phase, "p": p, "mixed_frac": round(frac, 4),
                       "loss": round(loss, 5), "acc": round(acc, 4), "sec": round(time.time() - s0, 3)}
                if tf_acc is not None and p > 0:
                    row["tf_acc"] = round(tf_acc, 4)
                events = self.sch.observe(self.step, acc)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                    for ev in events:
                        f.write(json.dumps(ev) + "\n")
                        log(f"[{self.tag}] {json.dumps(ev)}")
                if self.ctx is not None:
                    self.ctx.tick(1)
                if self.step % t["log_every"] == 0 or self.step == 1:
                    log(f"[{self.tag}] {json.dumps(row)}")
                    if self.tracker:
                        stage = Path(getattr(self.ctx, "stage_dir", self.out.parent)).name
                        self.tracker.log_values(f"curriculum/{stage}/thinker/{self.tag}", row)
                if t.get("eval_every") and self.step % t["eval_every"] == 0 and self.step < total_steps:
                    last_eval = evaluate(model)
                    self.history.append({"step": self.step, **last_eval})
                    log(f"[{self.tag}] eval {json.dumps(last_eval)}")
                    if self.tracker:
                        stage = Path(getattr(self.ctx, "stage_dir", self.out.parent)).name
                        self.tracker.log_values(f"curriculum/{stage}/thinker/{self.tag}/eval",
                                                {"step": self.step, **last_eval})
                if t.get("ckpt_every") and self.step % t["ckpt_every"] == 0:
                    self.save()
        finally:
            self.elapsed = base + time.time() - t0
            self.save()
        final = evaluate(model)
        self.history.append({"step": self.step, **final})
        if self.tracker:
            stage = Path(getattr(self.ctx, "stage_dir", self.out.parent)).name
            self.tracker.log_values(f"curriculum/{stage}/thinker/{self.tag}/eval",
                                    {"step": self.step, **final})
        self.save()
        return {"steps": self.step, "stopped": stopped, "seconds": round(self.elapsed, 1), "final": final,
                "history": self.history}


def bootstrap_ci(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> list[float]:
    """95% bootstrap CI of mean(a - b) (paired by seed); reuses connectome_experiment's helper."""
    try:
        from training.connectome_experiment import bootstrap_diff
        return bootstrap_diff(a, b, n=n, seed=seed)
    except Exception:
        d = np.array(a) - np.array(b)
        m = np.random.default_rng(seed).choice(d, size=(n, len(d)), replace=True).mean(axis=1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def build(kind: str, cfg: dict, seed: int, **over):
    mc = model_config(cfg, seed, **over)
    m = build_model(kind, mc)
    return m, param_report(m)
