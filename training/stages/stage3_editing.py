"""Stage 3 (editing, PLAN.md section 3).

Physical: BC (expert key-queue demos, clean action labelled) then PPO on edit-key sequences (Backspace/arrows, plus
the Fn chords Delete/Home/End when the env supports held-mode chords; unsupported ones are filtered and counted),
warm-started from stage 2's best checkpoint. Gate: >=95% correct physical edit keys.
Symbolic: ``training.edit_trainer.run_stage`` (lazy import; ``symbolic part pending`` if missing). Gate: >=95% of edits
reach the target within 1.5x the oracle's keystrokes (its ``first_round_ok``).
"""
from pathlib import Path

from training.stages import StageResult
from training.stages import physical_common as pcm

NAME = "stage3_editing"
STAGE = 3
RULES = {"edit_key_accuracy": ("edit_key_accuracy", "ge")}
SYM_KEY = "symbolic_within_1.5x_oracle"


def physical(ctx, stage_cfg: dict, smoke: bool, stage: int = STAGE, slip_p: float = 0.0, noise_eval: float = 0.0) -> dict:
    """Shared by stages 3 and 4 (``slip_p`` > 0 trains on slipped-then-corrected key strings)."""
    from training.tokens import token_splits
    pc = pcm.phys_cfg(stage_cfg, smoke)
    config = pcm.build_config(pc)
    tok = token_splits(**config["tokens"])
    include_fn = bool(pc.get("include_fn", True))
    train_s = pcm.EditSampler(tok["train"], include_fn, slip_p)
    held_s = pcm.EditSampler(tok["heldout"] or tok["train"], include_fn, slip_p)
    out = Path(ctx.stage_dir)
    warm = pcm.prev_checkpoint(ctx, stage) or stage_cfg.get("init_from")
    wrap = lambda env, i: pcm.KeyQueueWrapper(env, train_s, seed=ctx.seed * 100 + i)  # noqa: E731
    noise = pc["bc_noise"] if pc["bc_noise"] is not None else config["imitation"]["action_noise"]
    collect = lambda c, cfg, n, seed: pcm.collect_keys(c, cfg, train_s, n, seed, noise)  # noqa: E731
    phase = {"name": f"edit{stage}", "pool": ["a"], "bc_episodes": pc["bc_episodes"], "bc_epochs": pc["bc_epochs"],
             "ppo_timesteps": pc["ppo_timesteps"], "collect": collect, "wrapper": wrap}

    def on_phase(name, model, rec):
        if stage == 3:
            ev = pcm.eval_edit_keys(ctx, model, config, held_s, pc["eval_episodes"], seed=10_000 + ctx.seed)
            rec["eval"], score = ev, ev["edit_key_accuracy"]
        else:
            ev = pcm.eval_recovery(ctx, model, config, tok["heldout"] or tok["train"], pc["eval_episodes"],
                                   10_000 + ctx.seed, noise_eval)
            ev.update(pcm.eval_edit_keys(ctx, model, config, held_s, max(1, pc["eval_episodes"] // 2), 20_000 + ctx.seed))
            rec["eval"], score = ev, ev["recovery_exact"]
        return score if score is not None else 0.0

    log, cands, _ = pcm.train_phases(ctx, config, pc, [phase], warm, ctx.seed, out, on_phase)
    best, _s = pcm.pick_best(cands, out / "best_model.zip")
    return {"phases": log, "warm_start": warm, "best_checkpoint": best, "eval": log[-1]["eval"] if log else {}}


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    ph = physical(ctx, stage_cfg, smoke)
    ev = ph["eval"]
    metrics = {**ev, "phases": ph["phases"], "warm_start": ph["warm_start"], "best_checkpoint": ph["best_checkpoint"],
               "smoke": smoke}
    ok, detail = pcm.gate_check(ev, stage_cfg.get("gate", {}), RULES)
    artifacts = [ph["best_checkpoint"]] if ph["best_checkpoint"] else []
    if stage_cfg.get("symbolic", True):
        sym = pcm.run_symbolic(ctx, stage_cfg, smoke, STAGE)
        metrics["symbolic"] = sym
        if sym["status"] == "ok":
            sm = sym["metrics"]
            metrics[SYM_KEY] = sm.get("first_round_ok")
            g = {SYM_KEY: stage_cfg.get("gate", {}).get(SYM_KEY, 0.95)}
            sok, sdetail = pcm.gate_check(metrics, g, {SYM_KEY: (SYM_KEY, "ge")})
            detail.update(sdetail)
            ok = ok and sok
            artifacts += sym["artifacts"]
        else:
            detail[SYM_KEY] = {"ok": False, "value": None, "note": sym["status"]}
            ok = False
    passed = pcm.finalize_gate(metrics, ok, detail, stage_cfg, smoke)
    metrics["status"] = (f"edit_key_acc={ev.get('edit_key_accuracy')} unsupported={ev.get('edit_unsupported')} "
                         f"symbolic={metrics.get('symbolic', {}).get('status', 'off')} "
                         f"gate={metrics['gate_mode']} real_pass={ok}")
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=artifacts)
