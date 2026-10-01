"""Stage 2 (transcription, PLAN.md section 3): BC then PPO over ``datasets.transcription_pool`` levels
(words -> code_16 -> code_32 -> code_64 -> code_128 -> code_multi), warm-started from stage 1's best checkpoint.
Gate (held-out): >=90% exact on 3-8 char tokens and >=70% exact on 1-2 line code strings.
"""
from pathlib import Path

from training.stages import StageResult
from training.stages import physical_common as pcm

NAME = "stage2_transcription"
LEVELS = ("words", "code_16", "code_32", "code_64", "code_128", "code_multi")
RULES = {"token_exact": ("token_exact", "ge"), "code_exact": ("code_exact", "ge")}


def code_pools(seed: int) -> tuple[dict, dict, str]:
    """(train pool, held-out pool, held-out split name) from the dataset records; deterministic in ``seed``."""
    from training import datasets as ds
    recs = ds.load_all()
    splits = ds.split_records(recs, seed=seed)
    train = ds.transcription_pool(splits["train"], seed=seed)
    for name in ("test", "val"):
        held = ds.transcription_pool(splits[name], seed=seed) if splits[name] else {}
        if any(held.get(k) for k in LEVELS[1:]):
            return train, held, name
    return train, ds.transcription_pool(splits["train"], seed=seed + 1), "train(no held-out split available)"


def texts(pool: dict, level: str, max_lines: int = 10**9) -> list[str]:
    return [r["text"] for r in pool.get(level, []) if r["text"].count("\n") + 1 <= max_lines]


def heldout_code(held: dict, levels) -> list[str]:
    """1-2 line code strings across the active levels."""
    out = []
    for lv in levels:
        if lv != "words":
            out += texts(held, lv, max_lines=2)
    return out


def phases_for(pc: dict, tok_train: list[str], chars: list[str], train: dict, levels) -> list[dict]:
    ph = []
    for lv in levels:
        pool = (tok_train + chars) if lv == "words" else texts(train, lv)
        if not pool:
            continue
        code = lv != "words"
        ph.append({"name": lv, "pool": pool,
                   "bc_episodes": pc.get("bc_episodes_code", pc["bc_episodes"] // 4) if code else pc["bc_episodes"],
                   "bc_epochs": pc["bc_epochs"], "ppo_timesteps": pc["ppo_timesteps"]})
    return ph


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    from training.tokens import single_characters, token_splits
    pc = pcm.phys_cfg(stage_cfg, smoke)
    config = pcm.build_config(pc)
    levels = [lv for lv in pc.get("levels", LEVELS) if lv in LEVELS]
    tok = token_splits(**config["tokens"])
    train_pool, held_pool, held_name = code_pools(ctx.seed)
    code_eval = heldout_code(held_pool, levels)
    out = Path(ctx.stage_dir)
    warm = pcm.prev_checkpoint(ctx, 2) or stage_cfg.get("init_from")
    n_code = int(pc.get("eval_code_episodes", pc["eval_episodes"]))

    def on_phase(name, model, rec):
        te = pcm.eval_pool(ctx, model, config, tok["heldout"], pc["eval_episodes"], seed=10_000 + ctx.seed)
        ce = pcm.eval_pool(ctx, model, config, code_eval, n_code, seed=20_000 + ctx.seed) if code_eval else []
        rec["eval"] = {"token_exact": pcm.exact_rate(te), "code_exact": pcm.exact_rate(ce),
                       "token_cer": sum(e["cer"] for e in te) / max(1, len(te)),
                       "code_cer": (sum(e["cer"] for e in ce) / len(ce)) if ce else None,
                       "token_episodes": len(te), "code_episodes": len(ce)}
        return (rec["eval"]["token_exact"] or 0) + (rec["eval"]["code_exact"] or 0)

    phases = phases_for(pc, tok["train"], single_characters(tok["train"]), train_pool, levels)
    log, cands, _last = pcm.train_phases(ctx, config, pc, phases, warm, ctx.seed, out, on_phase)
    best, _s = pcm.pick_best(cands, out / "best_model.zip")
    ev = next((r["eval"] for r in log if best and Path(r["ckpt"]).read_bytes() == Path(best).read_bytes()),
              log[-1]["eval"] if log else {})
    metrics = {**ev, "phases": log, "warm_start": warm, "best_checkpoint": best, "heldout_code_split": held_name,
               "levels": levels, "smoke": smoke}
    ok, detail = pcm.gate_check(ev, stage_cfg.get("gate", {}), RULES)
    passed = pcm.finalize_gate(metrics, ok, detail, stage_cfg, smoke)
    metrics["status"] = (f"token_exact={ev.get('token_exact')} code_exact={ev.get('code_exact')} "
                         f"gate={metrics['gate_mode']} real_pass={ok}")
    return StageResult(metrics=metrics, gate_passed=passed, artifacts=[best] if best else [])
