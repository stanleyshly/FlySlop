"""Stage 9 (self-correction, PLAN.md section 3): write, run, observe, edit, run.

Generation policy ``G`` = stage 8 checkpoint (else stage 6); repair policy ``R`` starts from the stage 3-4 edit policy
(``repair_init``: ``auto`` = stage 4/3 symbolic ckpt of this run, else the newest under the runs directory, else a copy
of ``G``; ``generation`` forces the copy, which smoke uses so both models are the cheap ``gru`` kind; or a path).
``selfcorrect_train`` fine-tunes ``R`` on ``SelfCorrectEnv`` rollouts of TRAIN tasks only (dataset records with tests,
corrupted via ``error_inject`` or ``G``'s own failing attempts) by rejection-sampling fine-tuning (``method: rft``) or
REINFORCE with a baseline (``method: reinforce``). The held-out micro-suite is evaluation only.

Gate: pass@1 after up to ``max_rounds`` (3) repair rounds minus pass@1 with no repair, on the micro-suite, in mode
``gate_mode`` (``scratch``: G writes then R repairs; ``corrupt``: R repairs the reference with injected errors), must be
> ``gate.pass_at_1_gain`` (0.0 means any strict improvement, otherwise >=). Both modes are always reported, before and
after training. Smoke: SV capped (Verilator ~5 s/task), relaxed gate (NOT a real gate).
"""
import json
import time
from pathlib import Path

from training.stages import StageResult
from training.stages import physical_common as pcm
from training.stages.stage6_distill import merged_cfg
from training.stages.stage7_microprograms import select_tasks

NAME = "stage9_selfcorrect"


def gain_ok(improvement, thr: float) -> bool:
    if improvement is None:
        return False
    return improvement > 0 if thr <= 0 else improvement >= thr


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    import torch
    from training import distill
    from training import selfcorrect_train as sct
    from training import thinker_train_common as C
    sc = merged_cfg(stage_cfg, smoke)
    scc = {**sct.DEFAULT_SC, **sc.get("selfcorrect", {})}
    ov = dict(sc.get("overrides", {}))
    cfg = C.load_cfg(sc.get("thinker_config"), ov)
    C.guard_ram(ctx.ram_gb)
    torch.set_num_threads(4)
    gen_ckpt = sc.get("gen_ckpt") or sct.find_stage_ckpt(ctx, 8) or sct.find_stage_ckpt(ctx, 6)
    if not gen_ckpt:
        raise FileNotFoundError("stage 9 needs the stage 8/6 checkpoint (artifact ckpt.pt) or stage config gen_ckpt")
    ri = sc.get("repair_init", "auto")
    rep_ckpt = gen_ckpt if ri == "generation" else (ri if ri != "auto" else (sct.find_edit_ckpt(ctx) or gen_ckpt))
    gen, gblob = sct.load_thinker(gen_ckpt)
    repair, rblob = sct.load_thinker(rep_ckpt)
    tok = distill._tok(cfg)
    tcfg = {**cfg["train"], "max_len": scc["max_ctx"] + scc["max_new_edit"] + 2}
    workers = max(1, min(ctx.max_workers or 1, scc.get("workers") or 4))
    recs = distill.load_pairs("dataset", seed=ctx.seed)["train"]
    train_tasks = sct.build_train_tasks(recs, scc["lang"], scc["n_train_tasks"], ctx.seed, scc["max_code_chars"],
                                        verify=scc["verify_reference"], timeout=scc["run_timeout"])
    eval_tasks = select_tasks(sc.get("max_py"), sc.get("max_sv"))
    t0 = time.time()
    res = sct.run_selfcorrect(gen, repair, tok, tcfg, scc, ctx.stage_dir, train_tasks, eval_tasks, workers, ctx.seed, ctx)
    gate_mode = sc.get("gate_mode", "scratch")
    after, before = res["after"], res["before"]
    g = after[gate_mode]
    metrics = {"method": scc["method"], "gate_mode": gate_mode, "max_rounds": scc["max_rounds"],
               "no_repair_pass_at_1": g["no_repair_pass_at_1"], "repair_pass_at_1": g["repair_pass_at_1"],
               "improvement": g["improvement"], "before_training": before, "after_training": after,
               "warmup": res["warmup"], "train_history": res["train"]["history"], "train_tasks": len(train_tasks), "eval_tasks": len(eval_tasks),
               "gen_ckpt": gen_ckpt, "gen_kind": gblob["kind"], "repair_init": rep_ckpt, "repair_kind": rblob["kind"],
               "repair_init_source": ("generation ckpt" if rep_ckpt == gen_ckpt else "stage 3-4 edit policy"
                                      if ri == "auto" else "explicit path"),
               "solve_rate_first_iter": (res["train"]["history"] or [{}])[0].get("solve_rate"),
               "solve_rate_last_iter": (res["train"]["history"] or [{}])[-1].get("solve_rate"), "smoke": smoke,
               "seconds": round(time.time() - t0, 1)}
    thr = float(stage_cfg.get("gate", {}).get("pass_at_1_gain", 0.0))
    real_ok = gain_ok(g["improvement"], thr)
    detail = {"pass_at_1_gain": {"metric": f"{gate_mode}.improvement", "value": g["improvement"], "threshold": thr,
                                 "op": "gt" if thr <= 0 else "ge", "ok": real_ok}}
    passed = pcm.finalize_gate(metrics, real_ok, detail, sc, smoke)
    metrics["status"] = (f"{gate_mode}: no_repair={g['no_repair_pass_at_1']} repair@{scc['max_rounds']}={g['repair_pass_at_1']} "
                         f"gain={g['improvement']} method={scc['method']} gate={metrics['gate_mode']} real_pass={real_ok}")
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=[res["checkpoint"]])
