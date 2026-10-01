"""Pure stub stage module for orchestrator tests (the real stage modules are no longer stubs)."""
from training.stages import stub_run


def run(ctx, stage_cfg, smoke):
    return stub_run("stub", ctx, stage_cfg, smoke)
