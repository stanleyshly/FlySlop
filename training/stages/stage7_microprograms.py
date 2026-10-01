"""Stage 7 (autonomous micro-programs, PLAN.md section 3): pass@1 of the stage 6 policy on the held-out micro-suite
(Python + SV tasks with tests, greedy, one attempt), plus physical typing of the policy's own output on a subset.

* Checkpoint: ``init_from`` or the stage 6 artifact. It is loaded with ``selfcorrect_train.load_thinker`` (adapter
  for the distill ``ckpt.pt`` layout: ``model``/``kind``/``cfg``; ``eval_microsuite.load_policy`` expects
  ``state_dict`` and would fail on it) and driven through ``GenPolicy`` (``BOS prompt SEP -> code``, the distillation
  format; ``eval_microsuite.ThinkerPolicy`` feeds the stage 9 ``#TASK`` context, which stage 6 never saw).
* Optional ``finetune_on_failures`` (default off): greedy-generate on TRAIN dataset tasks, fine-tune (teacher forced,
  reference code) on the failures, and re-evaluate. The micro-suite is never trained on.
* Physical: ``physical_eval.type_tokens`` types the generated text of ``phys_tasks`` tasks (truncated to
  ``phys_max_chars``) with ``typist`` (``expert`` | ``auto`` = stage 2 PPO checkpoint if present | path) and reports the
  exact-match rate (typed editor state == symbolic result) and the contact-verified rate.
Gate: pass@1 >= ``pass_at_1_floor`` (and physical exact rate >= ``physical_exact_rate`` when configured).
Smoke: caps the SV task count (Verilator ~5 s/task), relaxed gate (NOT a real gate).

Full run: FLYSLOP_MAX_RAM_GB=4 uv run --extra training python -m training.curriculum --resume <run> (stage 7 config in curriculum.json)
"""
import json
import time
from pathlib import Path

from training.stages import StageResult
from training.stages import physical_common as pcm
from training.stages.stage6_distill import merged_cfg

NAME = "stage7_microprograms"


def select_tasks(max_py=None, max_sv=None) -> list[dict]:
    from training.microsuite import load_tasks
    py, sv = load_tasks("py"), load_tasks("sv")
    return (py if max_py is None else py[:max_py]) + (sv if max_sv is None else sv[:max_sv])


def per_lang(rep: dict) -> dict:
    return {k.split("=", 1)[1]: v["pass_at_1"] for k, v in rep["by_group"].items() if k.startswith("lang=") and k.count("=") == 1}


def per_difficulty(rep: dict) -> dict:
    return {k.split("=", 1)[1]: v["pass_at_1"] for k, v in rep["by_group"].items() if k.startswith("difficulty=") and k.count("=") == 1}


def resolve_typist(ctx, typist: str) -> str:
    if typist == "auto":
        return pcm.prev_checkpoint(ctx, 3) or "expert"   # stage 2 best_model.zip
    return typist


def physical_subset(policy, tok, tasks: list[dict], n: int, max_chars: int, typist: str, action_mode: str, seed: int,
                    typer=None) -> dict:
    """Type the policy's generated text for the first ``n`` tasks (py first, then sv) on the physical fly."""
    if typer is None:
        from training.physical_eval import type_tokens as typer
    order = sorted(tasks, key=lambda t: (t["lang"] != "py", t["difficulty"], t["lines"], t["id"]))
    rows = []
    for t in order:
        if len(rows) >= n:
            break
        text = tok.decode(policy.write(t))
        full_len = len(text)
        text = text[:max_chars]
        if not text.strip():
            continue                              # nothing generated: not counted as a typing attempt
        r = typer(text, typist=typist, action_mode=action_mode, seed=seed, max_ticks=None)
        rows.append({"task": t["id"], "lang": t["lang"], "chars": len(text), "truncated": full_len > len(text),
                     "exact": bool(r["exact"]), "text_match": bool(r["text_match"]),
                     "contact_verified": bool(r["contact_verified"]), "ticks": r["ticks"],
                     "wall_s": round(r["wall_s"], 2)})
    rate = lambda k: round(sum(x[k] for x in rows) / len(rows), 4) if rows else None  # noqa: E731
    return {"typist": typist, "n": len(rows), "physical_exact_rate": rate("exact"), "text_match_rate": rate("text_match"),
            "contact_verified_rate": rate("contact_verified"), "rows": rows}


def finetune_failures(model, tok, cfg, sc, ctx, workers: int, seed: int, ram_gb) -> dict:
    """Failure-driven SFT on TRAIN dataset tasks; returns counts (model is updated in place)."""
    from training import distill
    from training import selfcorrect_train as sct
    from training import thinker_train_common as C
    from training.selfcorrect_env import JudgeCache
    from training.tokenizer import normalize
    data = distill.load_pairs("dataset", seed=seed)
    tasks = sct.build_train_tasks(data["train"], sc.get("ft_lang", "py"), sc.get("ft_tasks", 32), seed)
    pol, judge = sct.GenPolicy(model, tok, sc.get("max_new", 256)), JudgeCache()
    fails = []
    for t in tasks:
        if not judge(t["lang"], tok.decode(pol.write(t)), t["tests"], sc.get("run_timeout", 20.0))["ok"]:
            fails.append(t)
    items = []
    for t in fails:
        e = distill.encode_pair(tok, t["prompt"], t["code"], cfg["train"]["max_len"])
        if e is not None:
            items.append(e)
    tcfg = cfg["train"]
    opt = C.make_optimizer(model, tcfg["lr"] * 0.3, tcfg["conn_lr"] * 0.3)
    for g in opt.param_groups:
        g["lr"] = g["base"]
    res = sct.sft(model, opt, items, sc.get("ft_steps", 20), min(tcfg["batch"], max(len(items), 1)), tcfg, seed, ctx)
    return {"train_tasks": len(tasks), "failures": len(fails), "trained_on": len(items), **res}


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    import torch
    from training import distill
    from training import eval_microsuite as em
    from training import selfcorrect_train as sct
    from training import thinker_train_common as C
    sc = merged_cfg(stage_cfg, smoke)
    C.guard_ram(ctx.ram_gb)
    torch.set_num_threads(4)
    ckpt = sc.get("init_from") or sct.find_stage_ckpt(ctx, 6)
    if not ckpt:
        raise FileNotFoundError("stage 7 needs the stage 6 checkpoint (artifact ckpt.pt) or stage config init_from")
    cfg = C.load_cfg(sc.get("thinker_config"))
    model, blob = sct.load_thinker(ckpt)
    tok = distill._tok(cfg)
    tasks = select_tasks(sc.get("max_py"), sc.get("max_sv"))
    workers = max(1, min(ctx.max_workers or 1, sc.get("workers", 4)))
    metrics = {"checkpoint": ckpt, "kind": blob["kind"], "smoke": smoke, "n_tasks": len(tasks), "workers": workers}
    max_new = int(sc.get("max_new", 256))

    def evaluate() -> dict:
        t0 = time.time()
        rep = em.evaluate(sct.GenPolicy(model, tok, max_new), tasks, tok, max_rounds=0, workers=workers,
                          run_timeout=sc.get("run_timeout", 20.0))
        ctx.tick(len(tasks))
        return {"pass_at_1": rep["pass_at_1"], "by_lang": per_lang(rep), "by_difficulty": per_difficulty(rep),
                "n": rep["n"], "eval_s": round(time.time() - t0, 1), "timing": rep["timing"]}

    ev = evaluate()
    metrics.update(pass_at_1=ev["pass_at_1"], pass_at_1_by_lang=ev["by_lang"], pass_at_1_by_difficulty=ev["by_difficulty"],
                   eval_s=ev["eval_s"], mean_task_s=ev["timing"]["mean_task_s"])
    artifacts = []
    if sc.get("finetune_on_failures"):
        ft = finetune_failures(model, tok, cfg, sc, ctx, workers, ctx.seed, ctx.ram_gb)
        ev2 = evaluate()
        ftp = sct.save_thinker(model, Path(ctx.stage_dir) / "ckpt_ft.pt")
        metrics["finetune"] = {**ft, "pass_at_1_before": ev["pass_at_1"], "pass_at_1_after": ev2["pass_at_1"],
                               "by_lang_after": ev2["by_lang"]}
        artifacts.append(ftp)
    phys = physical_subset(sct.GenPolicy(model, tok, max_new), tok, tasks, int(sc.get("phys_tasks", 3)),
                           int(sc.get("phys_max_chars", 60)), resolve_typist(ctx, sc.get("typist", "expert")),
                           sc.get("action_mode", "mn"), ctx.seed) if sc.get("phys_tasks", 3) else {"n": 0}
    ctx.tick(1)
    metrics["physical"] = {k: v for k, v in phys.items() if k != "rows"}
    metrics["physical_rows"] = phys.get("rows", [])
    metrics["physical_exact_rate"] = phys.get("physical_exact_rate")
    gate = stage_cfg.get("gate", {})
    rules = {"pass_at_1_floor": ("pass_at_1", "ge")}
    if "physical_exact_rate" in gate:
        rules["physical_exact_rate"] = ("physical_exact_rate", "ge")
    real_ok, detail = pcm.gate_check(metrics, gate, rules)
    passed = pcm.finalize_gate(metrics, real_ok, detail, sc, smoke)
    metrics["status"] = (f"pass@1={metrics['pass_at_1']} by_lang={json.dumps(metrics['pass_at_1_by_lang'])} "
                         f"physical_exact={metrics['physical_exact_rate']} (n={phys.get('n')}) "
                         f"gate={metrics['gate_mode']} real_pass={real_ok}")
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=artifacts)
