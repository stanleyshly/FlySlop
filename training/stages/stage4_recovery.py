"""Stage 4 (error recovery, PLAN.md section 3).

Physical: BC + PPO on slipped-then-corrected key strings (``backend.error_inject.corrupt_keys``: wrong neighbour key,
Backspace back to the slip, retype), warm-started from stage 3. Gate: >=90% recovery on 50 held-out physical episodes
(``physical_eval.type_tokens`` with motor noise to provoke slips and replanning recovery; final text must be exact).
Symbolic: ``edit_trainer.run_stage(stage=4)`` on corrupted buffers (lazy import; ``symbolic part pending`` if missing).
"""
from training.stages import StageResult
from training.stages import physical_common as pcm
from training.stages.stage3_editing import physical

NAME = "stage4_recovery"
STAGE = 4
RULES = {"recovery_exact": ("recovery_exact", "ge")}
SYM_KEY = "symbolic_replan_reach"


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    pc = pcm.phys_cfg(stage_cfg, smoke)
    ph = physical(ctx, stage_cfg, smoke, stage=STAGE, slip_p=float(pc.get("slip_p", 0.15)),
                  noise_eval=float(pc.get("eval_noise", 0.3)))
    ev = ph["eval"]
    metrics = {**ev, "phases": ph["phases"], "warm_start": ph["warm_start"], "best_checkpoint": ph["best_checkpoint"],
               "smoke": smoke}
    gate = dict(stage_cfg.get("gate", {}))
    ok, detail = pcm.gate_check(ev, gate, RULES)
    artifacts = [ph["best_checkpoint"]] if ph["best_checkpoint"] else []
    if stage_cfg.get("symbolic", True):
        sym = pcm.run_symbolic(ctx, stage_cfg, smoke, STAGE)
        metrics["symbolic"] = sym
        if sym["status"] == "ok":
            metrics[SYM_KEY] = sym["metrics"].get("replan_reach")
            sok, sdetail = pcm.gate_check(metrics, {SYM_KEY: gate.get(SYM_KEY, 0.90)}, {SYM_KEY: (SYM_KEY, "ge")})
            detail.update(sdetail)
            ok = ok and sok
            artifacts += sym["artifacts"]
        else:
            detail[SYM_KEY] = {"ok": False, "value": None, "note": sym["status"]}
            ok = False
    passed = pcm.finalize_gate(metrics, ok, detail, stage_cfg, smoke)
    metrics["status"] = (f"recovery_exact={ev.get('recovery_exact')} slip_eps={ev.get('slip_episodes')} "
                         f"symbolic={metrics.get('symbolic', {}).get('status', 'off')} "
                         f"gate={metrics['gate_mode']} real_pass={ok}")
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=artifacts)
