"""Stage 6 (distillation A->B->C, PLAN.md section 3): ``training.distill.run_stage`` (teacher forcing -> scheduled
sampling -> student-only) on oracle pairs (``data: auto`` falls back to dataset records).

Full run (``controls: true``): every kind x seed from ONE shared config (real, shuffled, random_sparse, frozen, gru);
gate = real mean token accuracy >= ``min_token_acc`` AND the paired-bootstrap 95% CI of (real - control) token
accuracy has a lower bound > 0 against BOTH shuffled and random_sparse. The stage artifact is the ``real`` seed-0
checkpoint of the controls grid (``controls/real_s<seed0>/ckpt.pt``) plus ``controls.md``.
Smoke (``smoke_stage`` merged in; default ``kind: gru``): one short run, relaxed gate (NOT a real gate);
``real_gate_passed`` records the honest outcome (token accuracy floor).

Offline full run (same code path):
    FLYSLOP_MAX_RAM_GB=4 uv run --extra training python -m training.curriculum --from-stage 6 --resume <run>
    uv run --extra training python -m training.distill --controls --seeds 0 1 2 3 4 --steps 20000 --out runs/distill/controls
"""
from pathlib import Path

from training import distill
from training.stages import StageResult
from training.stages import physical_common as pcm

NAME = "stage6_distill"
CONTROLS = ("shuffled", "random_sparse")


def merged_cfg(stage_cfg: dict, smoke: bool) -> dict:
    """Stage config with the ``smoke_stage`` block merged on top in smoke mode."""
    return {**stage_cfg, **(stage_cfg.get("smoke_stage", {}) if smoke else {})}


def beats_controls(summary: dict, controls=CONTROLS) -> tuple[bool, dict]:
    """Lower bound of ``real_minus_token_acc_ci95`` must be > 0 against every control present in the summary."""
    detail, ok = {}, bool(summary)
    for k in controls:
        ci = (summary.get(k) or {}).get("real_minus_token_acc_ci95")
        lo = ci[0] if ci else None
        detail[k] = {"ci95": ci, "ok": bool(lo is not None and lo > 0)}
        ok = ok and detail[k]["ok"]
    return ok, detail


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    sc = merged_cfg(stage_cfg, smoke)
    sc = {k: v for k, v in sc.items() if k != "smoke_stage"}
    floor = sc.get("min_token_acc", 0.2 if smoke else 0.5)
    sc["min_token_acc"] = floor
    out = Path(ctx.stage_dir)
    if sc.get("controls") and not smoke and "train.max_seconds" not in sc.get("overrides", {}) and "wall_s" in ctx.budget:
        # controls run kinds x seeds trainings from one shared config; split the wall budget between them
        n = len(sc.get("kinds", ["real", "shuffled", "random_sparse", "frozen", "gru"])) * len(sc.get("seeds", [0, 1, 2, 3, 4]))
        sc["overrides"] = {**sc.get("overrides", {}), "train.max_seconds": max(60.0, ctx.budget["wall_s"] / n - 30.0)}
    res = distill.run_stage(ctx, sc, smoke)
    m = dict(res.metrics)
    if sc.get("controls"):
        s = m.get("controls", {})
        beat, cdetail = beats_controls(s)
        seed0 = sc.get("seeds", [0])[0]
        real = s.get("real", {}).get("token_acc", {}).get("mean")
        acc_ok = real is not None and real >= floor
        detail = {"real_token_acc_floor": {"value": real, "threshold": floor, "ok": bool(acc_ok)}, **cdetail}
        real_ok = bool(beat and acc_ok)
        ckpt = out / f"real_s{seed0}" / "ckpt.pt"
        artifacts = ([str(ckpt)] if ckpt.exists() else []) + [a for a in res.artifacts if a not in {str(ckpt)}]
    else:
        acc = m.get("token_acc")
        real_ok = bool(res.gate_passed)
        detail = {"token_acc_floor": {"value": acc, "threshold": floor, "ok": real_ok},
                  "beats_controls": {"ok": None, "note": "controls not run (set controls: true for the real gate)"}}
        if not smoke:                       # a real gate needs the controls; a single run cannot pass it
            real_ok = False
        artifacts = list(res.artifacts)
    m["smoke"] = smoke
    passed = pcm.finalize_gate(m, real_ok, detail, sc, smoke)
    m["status"] = (f"kind={m.get('kind', 'controls' if sc.get('controls') else sc.get('kind'))} "
                   f"token_acc={m.get('token_acc')} gen_pass={m.get('gen_pass_rate')} gate={m['gate_mode']} real_pass={real_ok}")
    return StageResult(metrics=m, gate_passed=passed, artifacts=artifacts)
