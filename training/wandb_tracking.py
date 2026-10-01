"""Opt-in, scalar-only Weights & Biases tracking for FlySlop training.

No W&B import, run, network access, source capture, or artifact upload occurs unless a
project is supplied explicitly or through WANDB_PROJECT.
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path


_SENSITIVE = re.compile(r"prompt|text|traceback|command|cmd|checkpoint|artifact|path", re.I)


def options(project=None, entity=None, mode=None) -> dict:
    """Resolve explicit CLI values over the conventional W&B environment variables."""
    project = project or os.environ.get("WANDB_PROJECT")
    entity = entity or os.environ.get("WANDB_ENTITY")
    mode = mode or os.environ.get("WANDB_MODE")
    return {"project": project, "entity": entity, "mode": mode or "online"}


class WandbTracker:
    """Small failure-tolerant scalar logger. Local JSONL remains the durable source of record."""

    def __init__(self, run, run_id: str, mode: str):
        self.run, self.run_id, self.mode = run, run_id, mode
        self.disabled = False

    @classmethod
    def start(cls, run_dir: Path, project: str | None, entity: str | None, mode: str,
              config: dict, resume: bool = False):
        """Initialize only for an explicit project. Missing dependency is actionable; SDK errors degrade safely."""
        if not project:
            return None
        if mode not in {"online", "offline"}:
            raise ValueError("W&B mode must be 'online' or 'offline'")
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("W&B tracking was requested; install it with `uv sync --extra tracking`") from e

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "wandb_run.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        # Online resume continues the same remote run. Offline invocations remain separate local sessions.
        reuse = bool(resume and mode == "online" and meta.get("mode") == "online"
                     and meta.get("project") == project and meta.get("entity") == entity and meta.get("id"))
        run_id = meta["id"] if reuse else uuid.uuid4().hex[:12]
        try:
            run = wandb.init(
                project=project, entity=entity, mode=mode, id=run_id,
                resume="must" if reuse else "never", dir=str(run_dir), config=config,
                save_code=False,
                settings=wandb.Settings(console="off", disable_code=True, disable_git=True, init_timeout=30),
            )
        except Exception as e:  # tracking outages should not interrupt a requested training run
            print(f"[wandb] initialization failed; continuing with local logs only ({type(e).__name__})", file=sys.stderr,
                  flush=True)
            return None
        if run is None:
            print("[wandb] initialization returned no run; continuing with local logs only", file=sys.stderr, flush=True)
            return None
        meta_path.write_text(json.dumps({"id": run_id, "mode": mode, "project": project,
                                         "entity": entity}, indent=2) + "\n", encoding="utf-8")
        return cls(run, run_id, mode)

    @staticmethod
    def _flatten(value, prefix=""):
        out = {}
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                if _SENSITIVE.search(str(key)):
                    continue
                out.update(WandbTracker._flatten(child, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == value and abs(value) != float("inf"):
                out[prefix] = value
        elif isinstance(value, bool):
            out[prefix] = int(value)
        return out

    def log_event(self, event: dict) -> None:
        """Log numeric event fields and a small safe set of labels; never send payload text or errors."""
        event_name = str(event.get("event", "event"))
        ns = ["curriculum"]
        stage = event.get("stage", event.get("stage_id"))
        if stage is not None:
            ns.append(str(stage))
        for key in ("phase", "kind"):
            if isinstance(event.get(key), str):
                ns.append(event[key])
        ns.append(event_name)
        prefix = "/".join(ns)
        payload = {f"{prefix}/{key}": value for key, value in self._flatten(event).items()}
        # Restrict uploaded strings to low-risk status labels. Never send reason, stdout, or traceback text.
        for key in ("status", "gate_mode"):
            value = event.get(key)
            if isinstance(value, str):
                payload[f"{prefix}/{key}"] = value
        if event.get("event") == "stage_end" and event.get("status") == "error":
            # Do not upload exception messages or tracebacks, which can contain user data.
            payload["run/error"] = 1
            reason = str(event.get("reason", ""))
            error_type = reason.split(":", 1)[0].strip()
            if error_type and error_type.replace(".", "").replace("_", "").isalnum():
                payload["run/error_type"] = error_type
        self._send(payload)

    def log_values(self, namespace: str, values: dict) -> None:
        if self.disabled or self.run is None:
            return
        metrics = self._flatten(values)
        if not metrics:
            return
        prefix = f"{namespace}/" if namespace else ""
        payload = {prefix + key: value for key, value in metrics.items()}
        self._send(payload)

    def _send(self, payload: dict) -> None:
        if self.disabled or self.run is None:
            return
        try:
            # Let the SDK maintain its monotonic step across online resume.
            self.run.log(payload)
        except Exception as e:
            print(f"[wandb] logging failed; disabling tracking ({type(e).__name__})", file=sys.stderr, flush=True)
            self.disabled = True

    def finish(self, exit_code: int = 0) -> None:
        if self.run is None:
            return
        try:
            self.run.finish(exit_code=exit_code)
        except Exception as e:
            print(f"[wandb] finish failed ({type(e).__name__}); local logs are preserved", file=sys.stderr, flush=True)
        finally:
            self.run = None
