"""Curriculum stage runners (PLAN.md section 3).

Protocol: every ``stageN_*`` module exposes ``run(ctx, stage_cfg, smoke) -> StageResult``.
``ctx`` is a ``training.curriculum.StageContext``: call ``ctx.tick(steps)`` as training advances so
the orchestrator can enforce the per-stage step and wall-clock budgets (it raises BudgetExceeded).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageResult:
    metrics: dict = field(default_factory=dict)
    gate_passed: bool = False
    artifacts: list = field(default_factory=list)


def stub_run(name: str, ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    """Shared stub body. Gate passes iff ``stub`` is true. Test knobs: stub_steps, stub_sleep_s, stub_raise."""
    steps = int(stage_cfg.get("stub_steps", 1))
    sleep_s = float(stage_cfg.get("stub_sleep_s", 0.0))
    for _ in range(steps):
        ctx.tick(1)
        if sleep_s:
            time.sleep(sleep_s)
    if stage_cfg.get("stub_raise"):
        raise RuntimeError(f"injected failure in {name}")
    return StageResult(metrics={"status": "not implemented", "stage": name, "stub_steps": steps, "smoke": smoke},
                       gate_passed=bool(stage_cfg.get("stub", False)), artifacts=[])
