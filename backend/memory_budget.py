"""Project-wide RAM cap (default 4 GB). Every heavy component must go through this module.

Knob: env ``FLYSLOP_MAX_RAM_GB`` (float, default 4.0). Code may pass an explicit config value
to ``max_ram_bytes(config_gb=...)``; the env var wins when set.

macOS ignores RLIMIT_AS, so enforcement is (a) ``check_fits`` before big allocations and
(b) ``start_watchdog``, a daemon thread that aborts the process when process-tree RSS exceeds
the cap. No third-party dependencies (uses ``ps`` and ``resource``).
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import threading
import time

GB = 1024 ** 3
DEFAULT_GB = 4.0
ENV = "FLYSLOP_MAX_RAM_GB"


def max_ram_bytes(config_gb: float | None = None) -> int:
    env = os.environ.get(ENV)
    gb = float(env) if env else (float(config_gb) if config_gb is not None else DEFAULT_GB)
    if gb <= 0:
        raise ValueError(f"RAM cap must be positive, got {gb}")
    return int(gb * GB)


def _ps_table() -> dict[int, tuple[int, int]]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3:
            table[int(parts[0])] = (int(parts[1]), int(parts[2]) * 1024)
    return table


def current_rss(pid: int | None = None, tree: bool = False) -> int:
    """Resident set size in bytes of ``pid`` (default: this process); ``tree`` adds descendants."""
    pid = os.getpid() if pid is None else pid
    table = _ps_table()
    if pid not in table:
        return 0
    total, stack = table[pid][1], [pid]
    if tree:
        seen = {pid}
        while stack:
            cur = stack.pop()
            for child, (parent, rss) in table.items():
                if parent == cur and child not in seen:
                    seen.add(child); stack.append(child); total += rss
    return total


def peak_rss() -> int:
    """Peak RSS of this process so far (bytes). ru_maxrss is bytes on macOS, KiB on Linux."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v if sys.platform == "darwin" else v * 1024)


class OverBudget(MemoryError):
    pass


def check_fits(nbytes: int, label: str = "allocation", raise_error: bool = False,
               config_gb: float | None = None, headroom: float = 0.9) -> bool:
    """True if ``nbytes`` more on top of the tree's current RSS stays under ``headroom`` x cap."""
    cap = max_ram_bytes(config_gb)
    used = current_rss(tree=True)
    ok = used + nbytes <= cap * headroom
    if not ok:
        msg = (f"over RAM budget: {label} needs {nbytes / GB:.2f} GB, in use {used / GB:.2f} GB, "
               f"cap {cap / GB:.2f} GB ({ENV})")
        if raise_error:
            raise OverBudget(msg)
        print("[memory_budget] " + msg, file=sys.stderr, flush=True)
    return ok


def start_watchdog(interval_s: float = 1.0, fraction: float = 1.0, config_gb: float | None = None,
                   tree: bool = True) -> threading.Thread:
    """Abort (os._exit(87)) with a clear message if process-tree RSS exceeds fraction x cap."""
    limit = int(max_ram_bytes(config_gb) * fraction)

    def loop():
        while True:
            rss = current_rss(tree=tree)
            if rss > limit:
                print(f"\n[memory_budget] ABORT: RSS {rss / GB:.2f} GB exceeds cap {limit / GB:.2f} GB "
                      f"({ENV}); raise it or reduce workers/batch.", file=sys.stderr, flush=True)
                os._exit(87)
            time.sleep(interval_s)

    t = threading.Thread(target=loop, name="flyslop-ram-watchdog", daemon=True)
    t.start()
    return t


def worker_budget(per_worker_bytes: int, reserve_bytes: int = 0, config_gb: float | None = None,
                  max_workers: int | None = None) -> int:
    """How many parallel workers fit: floor((cap - reserve) / per_worker), at least 0."""
    avail = max_ram_bytes(config_gb) - int(reserve_bytes)
    n = max(0, int(avail // max(1, int(per_worker_bytes))))
    return min(n, max_workers) if max_workers is not None else n
