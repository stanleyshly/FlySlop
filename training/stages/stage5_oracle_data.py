"""Stage 5 (LLM oracle data, PLAN.md section 3): runs ``training.oracle_gen`` as a subprocess (its MLX teacher owns the
machine, hence the exclusive stage), polled so the stage budget can kill it. Outputs live under
``data/private/oracle/<run>/{pairs,train,val,test}.jsonl`` + ``manifest.json``.

Config (``stage_cfg["oracle"]``): pairs_min (10000), run, sources, holdout_sources, seed, spec_tokens (160),
code_tokens (384), neardup_test (0.7), neardup_accepted (0.9), time_budget_s, pairs_glob.
Gate: filtered (passed) pairs >= pairs_min, pass rate recorded. In smoke the gate is relaxed (``pairs_min_smoke``, labelled).
``neardup_*`` are passed to oracle_gen as --neardup-test/--neardup-accepted.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from training.stages import StageResult

NAME = "stage5_oracle_data"
DEFAULTS = {"pairs_min": 10000, "run": "curriculum", "sources": None, "holdout_sources": "", "seed": 0,
            "spec_tokens": 160, "code_tokens": 384, "neardup_test": 0.7, "neardup_accepted": 0.9,
            "time_budget_s": None, "pairs_glob": None, "standin": False, "limit": None, "poll_s": 1.0}
SMOKE = {"standin": True, "limit": 12, "pairs_min": 1, "time_budget_s": 90, "run": "curriculum_smoke"}


def oracle_cfg(stage_cfg: dict, smoke: bool) -> dict:
    o = {**DEFAULTS, **stage_cfg.get("oracle", {})}
    if smoke:
        o.update(SMOKE)
        o.update(stage_cfg.get("smoke_oracle", {}))
    return o


def count_pairs(root: Path, o: dict) -> tuple[int, int]:
    """(passed, attempts) over ``pairs_glob`` (default: this run's pairs.jsonl). Rows deduped by id."""
    files = sorted(Path(p) for p in __import__("glob").glob(o["pairs_glob"])) if o.get("pairs_glob") else \
        [root / o["run"] / "pairs.jsonl"]
    seen: dict[str, bool] = {}
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            seen[r["id"]] = seen.get(r["id"], False) or bool(r.get("passed"))
    return sum(seen.values()), len(seen)


def build_cmd(o: dict, time_budget_s: float | None, max_ram_gb: float | None) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "training.oracle_gen", "--run", str(o["run"]), "--seed", str(o["seed"]),
           "--spec-tokens", str(o["spec_tokens"]), "--code-tokens", str(o["code_tokens"])]
    if o.get("neardup_test") is not None:
        cmd += ["--neardup-test", str(o["neardup_test"])]
    if o.get("neardup_accepted") is not None:
        cmd += ["--neardup-accepted", str(o["neardup_accepted"])]
    if o["standin"]:
        cmd.append("--standin")
    if o["limit"] is not None:
        cmd += ["--limit", str(o["limit"])]
    if o["sources"]:
        cmd += ["--sources", o["sources"] if isinstance(o["sources"], str) else ",".join(o["sources"])]
    if o["holdout_sources"]:
        hs = o["holdout_sources"]
        cmd += ["--holdout-sources", hs if isinstance(hs, str) else ",".join(hs)]
    if time_budget_s is not None:
        cmd += ["--time-budget-s", f"{max(1.0, time_budget_s):.0f}"]
    if max_ram_gb:
        cmd += ["--max-ram-gb", str(max_ram_gb)]
    return cmd


def poll(ctx, proc: subprocess.Popen, poll_s: float) -> int:
    """Wait for ``proc``; tick the budget every poll and kill the child if it raises."""
    try:
        while proc.poll() is None:
            time.sleep(poll_s)
            ctx.tick(1)
    except BaseException:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc.returncode


def run(ctx, stage_cfg: dict, smoke: bool) -> StageResult:
    from training import datasets as ds
    o = oracle_cfg(stage_cfg, smoke)
    root = ds.PRIVATE / "oracle"
    have, attempts = count_pairs(root, o)
    tb = o["time_budget_s"]
    if "wall_s" in ctx.budget:
        remaining = max(1.0, ctx.budget["wall_s"] - ctx.elapsed - 5.0)
        tb = remaining if tb is None else min(tb, remaining)
    metrics = {"oracle": {k: v for k, v in o.items() if k != "poll_s"}, "pairs_before": have, "smoke": smoke}
    rc, cmd = None, None
    if have < o["pairs_min"] or smoke:
        cmd = build_cmd(o, tb, ctx.ram_gb)
        log = Path(ctx.stage_dir) / "oracle_gen.log"
        print(f"[stage5] oracle output is streaming to {log}", flush=True)
        with log.open("a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ds.ROOT if hasattr(ds, "ROOT") else "."))
            rc = poll(ctx, proc, float(o["poll_s"]))
        metrics["log"] = str(log)
        if rc != 0:
            raise RuntimeError(f"oracle_gen exited {rc}; see {log}")
    else:
        metrics["skipped_generation"] = "pairs_min already met"
    run_dir = root / o["run"]
    manifest = {}
    if (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text())
    passed_n, attempts = count_pairs(root, o)
    rate = passed_n / attempts if attempts else 0.0
    metrics.update({"filtered_pairs": passed_n, "attempts": attempts, "pass_rate": rate, "cmd": cmd,
                    "stopped": manifest.get("stopped"), "standin": bool(manifest.get("standin", o["standin"])),
                    "split_counts": {n: sum(1 for _ in (run_dir / f"{n}.jsonl").open()) if (run_dir / f"{n}.jsonl").exists() else 0
                                     for n in ("train", "val", "test")},
                    "gate_pairs_min": stage_cfg.get("gate", {}).get("filtered_pairs", o["pairs_min"])})
    metrics["gate_pairs_min"] = int(metrics["gate_pairs_min"])
    real_ok = passed_n >= metrics["gate_pairs_min"]
    metrics["real_gate_passed"] = bool(real_ok)
    if smoke and stage_cfg.get("smoke_gates", "relaxed") != "strict":
        passed = passed_n >= 1
        metrics["gate_mode"] = "relaxed (smoke, NOT a real gate): >=1 filtered pair"
    else:
        passed, metrics["gate_mode"] = real_ok, "strict"
    metrics["status"] = (f"filtered_pairs={passed_n}/{metrics['gate_pairs_min']} pass_rate={rate:.3f} "
                         f"gate={metrics['gate_mode']} real_pass={real_ok}")
    arts = [str(run_dir / n) for n in ("pairs.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "manifest.json")
            if (run_dir / n).exists()]
    return StageResult(metrics=metrics, gate_passed=bool(passed), artifacts=arts)
