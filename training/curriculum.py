"""One-command curriculum orchestrator (PLAN.md section 3 / P8).

    uv run --extra training python -m training.curriculum --config training/curriculum.json \
        [--smoke] [--resume RUN] [--from-stage N] [--advance-on-budget]

State machine persisted atomically at runs/curriculum/<ts>/state.json. Stage status:
pending -> running -> passed | gate_failed | budget_stopped | error; ``advanced`` marks a stage that
did not pass but was moved past under --advance-on-budget. Resume reruns the first stage that is
not passed/advanced/skipped (a stage left ``running`` by a killed process is rerun).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from backend import memory_budget
from training.common import ROOT, code_hash
from training.stages import StageResult
from training.wandb_tracking import WandbTracker, options as wandb_options

RUNS = ROOT / "runs" / "curriculum"
DONE = {"passed", "advanced", "skipped"}
CIRCUIT_IO = ROOT / "backend" / "connectome" / "circuit_io.py"


class BudgetExceeded(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind} budget exceeded: {detail}")
        self.kind = kind


class StageContext:
    """Handed to each stage: run dirs, seed, worker allowance, and budget accounting."""

    def __init__(self, run_dir: Path, stage_dir: Path, seed: int, budget: dict, exclusive: bool,
                 ram_gb: float, clock=time.monotonic, tracker=None):
        self.run_dir, self.stage_dir, self.seed = run_dir, stage_dir, seed
        self.budget, self.exclusive, self.ram_gb = budget, exclusive, ram_gb
        self.tracker = tracker
        self.steps = 0
        self._clock = clock
        self.started = clock()
        # exclusive stages (teacher) run alone: no concurrent workers
        self.max_workers = 0 if exclusive else max(1, memory_budget.worker_budget(
            int(0.5 * memory_budget.GB), reserve_bytes=int(1.0 * memory_budget.GB), config_gb=ram_gb, max_workers=8))

    @property
    def elapsed(self) -> float:
        return self._clock() - self.started

    def check(self) -> None:
        if "wall_s" in self.budget and self.elapsed > self.budget["wall_s"]:
            raise BudgetExceeded("wall_s", f"{self.elapsed:.1f}s > {self.budget['wall_s']}s")
        if "steps" in self.budget and self.steps > self.budget["steps"]:
            raise BudgetExceeded("steps", f"{self.steps} > {self.budget['steps']}")

    def tick(self, steps: int = 1) -> None:
        self.steps += steps
        self.check()


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_training_event(run_dir: Path, event: dict, tracker=None) -> None:
    """Append one durable, machine-readable progress/error event for this run."""
    path = Path(run_dir) / "training.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe(event), sort_keys=True, default=str, allow_nan=False) + "\n")
        f.flush()
    if tracker is not None:
        tracker.log_event(event)


def _sha(parts) -> str:
    d = hashlib.sha256()
    for p in parts:
        d.update(p if isinstance(p, bytes) else str(p).encode())
    return d.hexdigest()


def data_hash(paths: list[str], root: Path = ROOT) -> str | None:
    """Hash of file contents for files/dirs listed (dirs: relative names + sizes). None if none exist."""
    parts, found = [], False
    for rel in paths:
        p = root / rel
        if p.is_file():
            found = True
            parts += [rel, p.read_bytes()]
        elif p.is_dir():
            found = True
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                parts += [str(f.relative_to(root)), f.stat().st_size]
    return _sha(parts) if found else None


def circuit_hash() -> str | None:
    return _sha([CIRCUIT_IO.read_bytes()]) if CIRCUIT_IO.exists() else None


def config_hash(cfg: dict) -> str:
    return _sha([json.dumps(cfg, sort_keys=True)])


def effective_config(cfg: dict, smoke: bool) -> dict:
    """Apply the smoke profile: tiny budgets override each stage's budget."""
    cfg = json.loads(json.dumps(cfg))
    if smoke:
        for s in cfg["stages"]:
            s["budget"] = {**s.get("budget", {}), **cfg.get("smoke", {}).get("budget", {})}
            s.update(cfg.get("smoke", {}).get("stage_overrides", {}).get(str(s["id"]), {}))
            s.setdefault("smoke_gates", cfg.get("smoke", {}).get("gates", "relaxed"))  # smoke gates are not real gates
    return cfg


def new_state(cfg: dict, smoke: bool, args: dict) -> dict:
    return {"version": 1, "created": time.time(), "smoke": smoke, "args": args, "status": "running",
            "config_hash": config_hash(cfg), "config": cfg,
            "stages": [{"id": s["id"], "name": s["name"], "status": "pending", "attempts": 0} for s in cfg["stages"]]}


def _find_run(spec: str, runs: Path = RUNS) -> Path:
    p = Path(spec)
    if not p.is_absolute() and not (p / "state.json").exists():
        p = runs / spec
    if not (p / "state.json").exists():
        raise SystemExit(f"no state.json under {p}")
    return p


def run_stage(state: dict, entry: dict, scfg: dict, run_dir: Path, smoke: bool, ram_gb: float, seed: int,
              tracker=None) -> None:
    """Run one stage; mutate ``entry`` with status/manifest. Caller persists state."""
    stage_dir = run_dir / f"stage{scfg['id']}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    entry.pop("traceback", None)
    ctx = StageContext(run_dir, stage_dir, seed, scfg.get("budget", {}), bool(scfg.get("exclusive")), ram_gb,
                       tracker=tracker)
    t0 = time.time()
    result, reason = StageResult(), None
    append_training_event(run_dir, {"event": "stage_start", "stage": scfg["id"], "name": scfg["name"],
                                   "time": t0, "seed": seed}, tracker)
    print(f"[curriculum] stage {scfg['id']} {scfg['name']} started", flush=True)
    try:
        mod = importlib.import_module(scfg["module"])
        result = mod.run(ctx, scfg, smoke)
        ctx.check()  # a stage that never ticked still gets its wall budget checked
        entry["status"] = "passed" if result.gate_passed else "gate_failed"
    except BudgetExceeded as e:
        entry["status"], reason = "budget_stopped", str(e)
    except Exception as e:  # noqa: BLE001 - recorded, run is resumable
        entry["status"], reason = "error", "".join(traceback.format_exception_only(type(e), e)).strip()
        entry["traceback"] = traceback.format_exc()
        print(entry["traceback"], file=sys.stderr, flush=True)
    entry["reason"] = reason
    entry["metrics"] = result.metrics
    entry["gate_passed"] = bool(result.gate_passed) and entry["status"] == "passed"
    entry["artifacts"] = list(result.artifacts)
    manifest = {"stage": scfg["id"], "name": scfg["name"], "code_hash": code_hash(),
                "data_hash": data_hash(state["config"].get("data_files", [])), "circuit_hash": circuit_hash(),
                "config": scfg, "gate": scfg.get("gate"), "seed": seed, "exclusive": bool(scfg.get("exclusive")),
                "smoke": smoke, "started": t0, "time_s": time.time() - t0, "steps": ctx.steps,
                "peak_rss_bytes": memory_budget.peak_rss(), "status": entry["status"], "reason": reason,
                "metrics": result.metrics, "gate_passed": entry["gate_passed"]}
    atomic_write_json(stage_dir / "manifest.json", manifest)
    entry["manifest"] = str((stage_dir / "manifest.json").relative_to(run_dir))
    append_training_event(run_dir, {"event": "stage_end", "stage": scfg["id"], "name": scfg["name"],
                                   "time": time.time(), "time_s": manifest["time_s"], "steps": ctx.steps,
                                   "status": entry["status"], "reason": reason, "metrics": result.metrics,
                                   "traceback": entry.get("traceback")}, tracker)


def drive(run_dir: Path, state: dict, smoke: bool, advance_on_budget: bool, from_stage: int | None = None,
          ram_gb: float = 4.0, tracker=None) -> dict:
    """Advance the state machine until done or stopped; state.json is rewritten at every transition."""
    cfg, path = state["config"], run_dir / "state.json"
    if from_stage is not None:
        for e in state["stages"]:
            if e["id"] < from_stage and e["status"] not in DONE:
                e["status"] = "skipped"
    state["status"] = "running"
    atomic_write_json(path, state)
    for scfg, entry in zip(cfg["stages"], state["stages"]):
        if entry["status"] in DONE:
            continue
        entry["status"], entry["attempts"] = "running", entry["attempts"] + 1
        atomic_write_json(path, state)
        run_stage(state, entry, scfg, run_dir, smoke, ram_gb, cfg.get("seed", 0) + scfg["id"], tracker)
        if entry["status"] == "budget_stopped" and advance_on_budget:
            entry["status"] = "advanced"
        state["current"] = scfg["id"]
        atomic_write_json(path, state)
        print(f"[curriculum] stage {scfg['id']} {scfg['name']}: {entry['status']}"
              f" | {entry['metrics'].get('status', '')} | {entry.get('reason') or ''}", flush=True)
        if entry["status"] not in DONE:
            state["status"] = "stopped"
            state["stop_reason"] = f"stage {scfg['id']} {entry['status']}: {entry.get('reason') or 'gate not met'}"
            atomic_write_json(path, state)
            return state
    state["status"] = "complete"
    atomic_write_json(path, state)
    return state


def resolve_ram_gb(cli: float | None, cfg: dict, env: dict | None = None) -> float:
    """RAM cap precedence: ``--ram-gb`` > env ``FLYSLOP_MAX_RAM_GB`` (when set) > config ``ram_gb`` > default."""
    if cli is not None:
        return float(cli)
    raw = (os.environ if env is None else env).get(memory_budget.ENV)
    if raw:
        return float(raw)
    return float(cfg.get("ram_gb", memory_budget.DEFAULT_GB))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "training" / "curriculum.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", metavar="RUN")
    ap.add_argument("--from-stage", type=int)
    ap.add_argument("--advance-on-budget", action="store_true")
    ap.add_argument("--ram-gb", type=float, help="RAM cap in GB; beats env FLYSLOP_MAX_RAM_GB, which beats config ram_gb")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    help="connectome policy device for physical stages (auto uses available hardware)")
    ap.add_argument("--bc-workers", type=int,
                    help="override behavior-cloning data-collection workers for physical stages")
    ap.add_argument("--n-envs", type=int,
                    help="physical environment subprocesses (PPO rollout size is preserved when this changes)")
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--wandb-project", help="opt into scalar-only W&B logging (also accepts WANDB_PROJECT)")
    ap.add_argument("--wandb-entity", help="W&B entity (also accepts WANDB_ENTITY)")
    ap.add_argument("--wandb-mode", choices=("online", "offline"), help="defaults to online when a project is set")
    a = ap.parse_args(argv)
    runs = Path(a.runs_dir)

    if a.resume:
        run_dir = _find_run(a.resume, runs)
        state = json.loads((run_dir / "state.json").read_text())
        cfg, smoke = state["config"], state["smoke"]
    else:
        smoke = a.smoke
        cfg = effective_config(json.loads(Path(a.config).read_text()), smoke)
        run_dir = runs / time.strftime("%Y%m%d-%H%M%S")
        state = new_state(cfg, smoke, {"advance_on_budget": a.advance_on_budget, "from_stage": a.from_stage})
    if a.device:
        for stage in cfg.get("stages", []):
            if "physical" in stage:
                stage["physical"]["device"] = a.device
            if "smoke_physical" in stage:
                stage["smoke_physical"]["device"] = a.device
        state["config"] = cfg
        state["config_hash"] = config_hash(cfg)
        state.setdefault("args", {})["device"] = a.device
    if a.bc_workers is not None:
        if a.bc_workers < 1:
            ap.error("--bc-workers must be at least 1")
        for stage in cfg.get("stages", []):
            if "physical" in stage:
                stage["physical"]["bc_workers"] = a.bc_workers
            if "smoke_physical" in stage:
                stage["smoke_physical"]["bc_workers"] = a.bc_workers
        state["config"] = cfg
        state["config_hash"] = config_hash(cfg)
        state.setdefault("args", {})["bc_workers"] = a.bc_workers
    if a.n_envs is not None:
        if a.n_envs < 1:
            ap.error("--n-envs must be at least 1")
        for stage in cfg.get("stages", []):
            for key in ("physical", "smoke_physical"):
                pc = stage.get(key)
                if pc is not None:
                    pc["rollout_n_envs"] = int(pc.get("n_envs", 1))
                    pc["n_envs"] = a.n_envs
                    pc["preserve_rollout"] = True
        state["config"] = cfg
        state["config_hash"] = config_hash(cfg)
        state.setdefault("args", {})["n_envs"] = a.n_envs
    ram_gb = resolve_ram_gb(a.ram_gb, cfg)  # --ram-gb > env FLYSLOP_MAX_RAM_GB > config ram_gb
    os.environ[memory_budget.ENV] = str(ram_gb)
    memory_budget.start_watchdog(config_gb=ram_gb)
    print(f"[curriculum] run={run_dir} smoke={smoke} ram_gb={ram_gb}", flush=True)
    wo = wandb_options(a.wandb_project, a.wandb_entity, a.wandb_mode)
    tracker = WandbTracker.start(run_dir, wo["project"], wo["entity"], wo["mode"],
                                 {"run_type": "curriculum", "seed": cfg.get("seed", 0), "smoke": smoke,
                                  "ram_gb": ram_gb, "stage_ids": [s["id"] for s in cfg["stages"]],
                                  "stage_names": [s["name"] for s in cfg["stages"]], "curriculum": cfg},
                                 resume=bool(a.resume))
    exit_code = 1
    try:
        state = drive(run_dir, state, smoke, a.advance_on_budget or state["args"].get("advance_on_budget", False),
                      a.from_stage, ram_gb, tracker)
        print(f"[curriculum] {state['status']}" + (f": {state['stop_reason']}" if state.get("stop_reason") else ""),
              flush=True)
        exit_code = 0 if state["status"] == "complete" else 1
        return exit_code
    finally:
        if tracker:
            tracker.finish(exit_code)


if __name__ == "__main__":
    sys.exit(main())
