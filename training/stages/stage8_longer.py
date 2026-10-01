"""Stage 8 (longer programs, PLAN.md section 3): continue the stage 6 distillation with a length curriculum.

``levels`` (config) is an ordered list: reference length caps (``max_lines`` 1, 2, ... 5) and then feature levels
(``feature``: ``branch`` | ``loop`` | ``multi_fn``: programs with if/else, loops, or several functions/tasks, up to
``max_lines`` lines). For each level: train ``steps_per_level`` more steps (``distill.train_kind`` resumed from the previous
checkpoint, on the training pairs that fit the level; feature levels also replay the <=5-line pairs), then measure the pass
rate on the held-out micro-suite tasks of that level (length levels: reference lines <= cap; feature levels: tasks
with the feature). A level with no micro-suite tasks falls back to the held-out dataset (val) pairs of that level
(``source`` is recorded) and to a vacuous pass if there are none. The curriculum advances a level only while the pass
rate is >= ``gate.pass_rate_floor``; up to ``max_attempts`` extra training rounds are spent on a level that misses it,
then the stage stops there. Gate = all levels passed. Reports pass rate against length
(``pass_rate_by_lines``: latest result per micro-suite task, bucketed by reference lines).
Smoke: ``smoke_stage`` (gru, few levels/steps), SV excluded (Verilator ~5 s/task), relaxed gate (NOT a real gate).
"""
import ast
import re
import shutil
import time
from pathlib import Path

from training.stages import StageResult
from training.stages import physical_common as pcm
from training.stages.stage6_distill import merged_cfg

NAME = "stage8_longer"
DEFAULT_LEVELS = [{"name": f"lines<={n}", "max_lines": n} for n in (1, 2, 3, 4, 5)] + [
    {"name": "branches", "feature": "branch", "max_lines": 20}, {"name": "loops", "feature": "loop", "max_lines": 20},
    {"name": "multi_function", "feature": "multi_fn", "max_lines": 40}]


def n_lines(code: str) -> int:
    return len(code.strip().splitlines())


def features(code: str, lang: str) -> set:
    """{'branch','loop','multi_fn'} present in ``code`` (py via ast, sv via keywords)."""
    out = set()
    if lang == "py":
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return out
        nodes = list(ast.walk(tree))
        if any(isinstance(n, (ast.If, ast.IfExp, getattr(ast, "Match", ast.If))) for n in nodes):
            out.add("branch")
        if any(isinstance(n, (ast.For, ast.While, ast.AsyncFor)) for n in nodes):
            out.add("loop")
        if sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes) >= 2:
            out.add("multi_fn")
    else:
        if re.search(r"\b(if|case|casez|casex)\b|\?", code):
            out.add("branch")
        if re.search(r"\b(for|while|forever|repeat|generate)\b", code):
            out.add("loop")
        if len(re.findall(r"\b(module|function|task)\b", code)) >= 2:
            out.add("multi_fn")
    return out


def level_match(level: dict, code: str, lang: str, lines: int | None = None) -> bool:
    lines = n_lines(code) if lines is None else lines
    if lines > level.get("max_lines", 10**9):
        return False
    return level.get("feature") is None or level["feature"] in features(code, lang)


def level_train_pairs(level: dict, pairs: list[dict], replay_lines: int = 5) -> list[dict]:
    """Training pairs for a level: length levels are cumulative (<= cap); feature levels = feature pairs + short replay."""
    if level.get("feature"):
        return [p for p in pairs if level_match(level, p["code"], p["lang"]) or n_lines(p["code"]) <= replay_lines]
    return [p for p in pairs if level_match(level, p["code"], p["lang"])]


def level_eval_tasks(level: dict, tasks: list[dict]) -> list[dict]:
    return [t for t in tasks if level_match(level, t["code"], t["lang"], t["lines"])]


def pass_rate_pairs(model, tok, pairs: list[dict], max_new: int, cap: int, timeout: float = 10.0):
    """Greedy pass rate on dataset pairs (py only, tests present, at most ``cap``): (rate or None, n)."""
    from training import selfcorrect_train as sct
    from training.selfcorrect_env import JudgeCache
    ps = [p for p in pairs if p["lang"] == "py" and p.get("tests")][:cap]
    pol, judge = sct.GenPolicy(model, tok, max_new), JudgeCache()
    ok = sum(bool(judge(p["lang"], tok.decode(pol.write({"prompt": p["prompt"]})), p["tests"], timeout)["ok"]) for p in ps)
    return (round(ok / len(ps), 4) if ps else None), len(ps)


def pass_by_lines(rows_by_task: dict) -> dict:
    by = {}
    for r in rows_by_task.values():
        by.setdefault(r["lines"], []).append(bool(r["ok"]))
    return {str(k): {"n": len(v), "pass_rate": round(sum(v) / len(v), 4)} for k, v in sorted(by.items())}


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    import torch
    from training import distill
    from training import eval_microsuite as em
    from training import selfcorrect_train as sct
    from training import thinker_train_common as C
    from training.stages.stage7_microprograms import select_tasks
    sc = merged_cfg(stage_cfg, smoke)
    ov = dict(sc.get("overrides", {}))
    if smoke:
        ov = {"model.k_steps": 2, "train.batch": 8, "train.max_len": 96, "train.eval_pairs": 8, "train.gen_pairs": 4,
              "train.gen_max_new": 48, "train.eval_every": 0, "train.ckpt_every": 0, "train.log_every": 5, **ov}
    cfg = C.load_cfg(sc.get("thinker_config"), ov)
    cfg["train"]["max_seconds"] = 1e9      # Trainer.elapsed carries over on resume; ctx.tick enforces the stage wall budget
    C.guard_ram(ctx.ram_gb)
    torch.set_num_threads(4)
    init = sc.get("init_from") or sct.find_stage_ckpt(ctx, 6)
    if not init:
        raise FileNotFoundError("stage 8 needs the stage 6 checkpoint (artifact ckpt.pt) or stage config init_from")
    blob = torch.load(init, map_location="cpu", weights_only=False)
    kind = blob["kind"]
    levels = sc.get("levels", DEFAULT_LEVELS)
    floor = float(stage_cfg.get("gate", {}).get("pass_rate_floor", 0.0))
    steps_per_level, max_attempts = int(sc.get("steps_per_level", 200)), int(sc.get("max_attempts", 1))
    data = distill.load_pairs(sc.get("data", cfg["distill"]["data"]), cfg["distill"]["min_oracle_pairs"], ctx.seed)
    tok = distill._tok(cfg)
    tasks = select_tasks(sc.get("max_py"), sc.get("max_sv"))
    workers = max(1, min(ctx.max_workers or 1, sc.get("workers", 4)))
    out = Path(ctx.stage_dir)
    resume, step, latest, log, reached = init, int(blob.get("step", 0)), {}, [], 0
    del blob
    stopped = None
    for li, level in enumerate(levels):
        train_pairs = level_train_pairs(level, data["train"])
        rec = {"level": li, "name": level.get("name", f"L{li}"), "max_lines": level.get("max_lines"),
               "feature": level.get("feature"), "n_train_pairs": len(train_pairs), "attempts": 0, "trained_steps": 0}
        passed = False
        for attempt in range(max(1, max_attempts)):
            rec["attempts"] += 1
            t0 = time.time()
            if train_pairs and steps_per_level > 0:
                ld = out / f"level{li}"
                d = {"train": train_pairs, "val": [p for p in data["val"] if level_match(level, p["code"], p["lang"])] or data["val"],
                     "test": [], "label": data["label"] + f" | level {rec['name']}"}
                res = distill.train_kind(cfg, kind, ctx.seed, ld, d, step + steps_per_level, resume=resume, ctx=ctx,
                                         ram_gb=ctx.ram_gb)
                step, resume = res["steps"], str(ld / "ckpt.pt")
                rec["trained_steps"] += steps_per_level
                rec["train_final"] = {k: res["final"].get(k) for k in ("token_acc", "gen_pass_rate", "gen_parse_rate")}
            model, _ = sct.load_thinker(resume)
            et = level_eval_tasks(level, tasks)
            if et:
                rep = em.evaluate(sct.GenPolicy(model, tok, sc.get("max_new", 256)), et, tok, max_rounds=0, workers=workers,
                                  run_timeout=sc.get("run_timeout", 20.0))
                rate, src, n = rep["pass_at_1"], "micro", rep["n"]
                for r in rep["rows"]:
                    latest[r["id"]] = r
            else:
                held = [p for p in data["val"] if level_match(level, p["code"], p["lang"])]
                rate, n = pass_rate_pairs(model, tok, held, sc.get("max_new", 256), int(sc.get("val_cap", 20)))
                src = "val" if n else "none"
            del model
            ctx.tick(max(1, n))
            rec.update(pass_rate=rate, n_eval=n, source=src, level_s=round(time.time() - t0, 1))
            tracker = getattr(ctx, "tracker", None)
            if tracker is not None:
                tracker.log_values(f"curriculum/{Path(ctx.stage_dir).name}/thinker/level_{li}", rec)
            passed = rate is None or rate >= floor          # nothing to measure: vacuous pass, flagged by source
            if passed:
                break
        rec["advanced"] = passed
        log.append(rec)
        if not passed:
            stopped = f"level {rec['name']}: pass_rate {rec['pass_rate']} < floor {floor}"
            break
        reached = li + 1
        if li > 0 and (out / f"level{li - 1}").exists():
            shutil.rmtree(out / f"level{li - 1}", ignore_errors=True)    # keep only the newest level checkpoint
    final = out / "ckpt.pt"
    if resume and Path(resume).exists() and Path(resume) != Path(init):
        shutil.copy(resume, final)
    artifacts = [str(final)] if final.exists() else []
    metrics = {"levels": log, "levels_reached": reached, "levels_total": len(levels), "floor": floor, "kind": kind,
               "pass_rate_by_lines": pass_by_lines(latest), "init_from": init, "stopped": stopped, "smoke": smoke,
               "total_steps": step, "data": data["label"]}
    real_ok = reached == len(levels)
    detail = {"all_levels_above_floor": {"value": reached, "threshold": len(levels), "ok": real_ok, "floor": floor}}
    passed = pcm.finalize_gate(metrics, real_ok, detail, sc, smoke)
    metrics["status"] = (f"levels {reached}/{len(levels)} " + " ".join(f"{r['name']}={r['pass_rate']}" for r in log)
                         + f" gate={metrics['gate_mode']} real_pass={real_ok}")
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=artifacts)
