"""Stage 1 (key control, PLAN.md section 3): BC then PPO over growing key pools (8 home-row keys -> 26 letters -> all
characters). Gate (held-out episodes, env eval): >=95% single-key presses correct and <=1% unintended characters.
"""
from pathlib import Path

from training.stages import StageResult
from training.stages import physical_common as pcm

NAME = "stage1_keys"
HOME_ROW = "asdfjkl;"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
RULES = {"key_accuracy": ("key_accuracy", "ge"), "unintended_rate_max": ("unintended_rate", "le")}


def phases_for(pc: dict, full: list[str]) -> list[dict]:
    pools = [("keys8", sorted(HOME_ROW)), ("keys26", sorted(LETTERS)), ("keys_full", list(full))]
    return [{"name": n, "pool": p, "bc_episodes": pc["bc_episodes"], "bc_epochs": pc["bc_epochs"],
             "ppo_timesteps": pc["ppo_timesteps"]} for n, p in pools]


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    from training.tokens import single_characters, token_splits
    pc = pcm.phys_cfg(stage_cfg, smoke)
    config = pcm.build_config(pc)
    full = single_characters(token_splits(**config["tokens"])["train"])
    out = Path(ctx.stage_dir)
    warm = pcm.prev_checkpoint(ctx, 1) or stage_cfg.get("init_from")

    def on_phase(name, model, rec):
        eps = pcm.eval_pool(ctx, model, config, full, pc["eval_episodes"], seed=10_000 + ctx.seed)
        rec["eval"] = pcm.key_metrics(eps)
        return rec["eval"]["key_accuracy"] - rec["eval"]["unintended_rate"]

    log, cands, _last = pcm.train_phases(ctx, config, pc, phases_for(pc, full), warm, ctx.seed, out, on_phase)
    best, _score = pcm.pick_best(cands, out / "best_model.zip")
    ev = next((r["eval"] for r in log if r["ckpt"] and best and Path(r["ckpt"]).read_bytes() == Path(best).read_bytes()),
              log[-1]["eval"] if log else {})
    metrics = {**ev, "phases": log, "warm_start": warm, "best_checkpoint": best, "smoke": smoke,
               "status": f"key_acc={ev.get('key_accuracy')} unintended={ev.get('unintended_rate')}"}
    ok, detail = pcm.gate_check(ev, stage_cfg.get("gate", {}), RULES)
    passed = pcm.finalize_gate(metrics, ok, detail, stage_cfg, smoke)
    metrics["status"] += f" gate={metrics['gate_mode']} real_pass={ok}"
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=[best] if best else [])
