"""Sandboxed execution judge for generated Python and SystemVerilog.

Result: ``{ok, stage: parse|elab|test, stdout_tail, error, duration}``.
``stage`` is the stage reached: the failing stage, or ``test`` when everything passed
(``elab`` when SV passed elaboration but no simulator is installed; see ``available_tools``).

Python runs in a fresh interpreter with a temp cwd, a scrubbed environment, a wall-clock timeout
(process group killed), CPU/file-size/memory rlimits where the OS honours them, and a
``sitecustomize`` that disables sockets.  This is a best-effort guard for our own generated code,
not a security boundary against a determined adversary.
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TAIL_CHARS = 2000

_NO_NET = '''\
import socket as _s

def _blocked(*a, **k):
    raise OSError("network disabled by FlySlop exec_judge")

class _NoSocket(_s.socket):
    def __init__(self, *a, **k):
        raise OSError("network disabled by FlySlop exec_judge")

_s.socket = _NoSocket
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
_s.gethostbyname = _blocked
_s.socketpair = _blocked
'''


def available_tools() -> dict[str, str | None]:
    """Which SV simulators exist on PATH (nothing is installed by this module)."""
    return {"iverilog": shutil.which("iverilog"), "vvp": shutil.which("vvp"), "verilator": shutil.which("verilator")}


def _result(ok: bool, stage: str, stdout: str = "", error: str = "", start: float = 0.0, **extra: Any) -> dict[str, Any]:
    return {"ok": ok, "stage": stage, "stdout_tail": stdout[-TAIL_CHARS:], "error": error[-TAIL_CHARS:],
            "duration": round(time.monotonic() - start, 4), **extra}


def _limits(mem_bytes: int | None, cpu_s: int, fsize: int = 16 << 20):
    def apply() -> None:
        os.setsid()
        for res, val in ((resource.RLIMIT_CPU, cpu_s), (resource.RLIMIT_FSIZE, fsize),
                         (resource.RLIMIT_CORE, 0), (resource.RLIMIT_AS, mem_bytes),
                         (resource.RLIMIT_DATA, mem_bytes)):
            if val is None:
                continue
            try:  # macOS refuses some limits; skip silently
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass
    return apply


def _run(cmd: list[str], cwd: str, timeout: float, env: dict[str, str], mem_bytes: int | None):
    """Run cmd; returns (returncode|None on timeout, stdout, stderr)."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, errors="replace",
                         preexec_fn=_limits(mem_bytes, int(timeout) + 2))
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            p.kill()
        out, err = p.communicate()
        return None, out or "", err or ""


def _env(extra_path: str | None = None) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "HOME": tempfile.gettempdir()}
    if extra_path:
        env["PYTHONPATH"] = extra_path
    return env


def run_python(code: str, tests: str = "", timeout: float = 10.0, mem_mb: int | None = 1024) -> dict[str, Any]:
    """Execute ``code`` then ``tests`` in one module. Pass = exit code 0 within the timeout."""
    start = time.monotonic()
    try:
        compile(code + "\n\n" + tests, "<candidate>", "exec")
    except SyntaxError as e:
        return _result(False, "parse", error=f"SyntaxError: {e}", start=start)
    with tempfile.TemporaryDirectory(prefix="flyslop-exec-") as tmp:
        site = Path(tmp) / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(_NO_NET)
        work = Path(tmp) / "work"
        work.mkdir()
        (work / "main.py").write_text(code + "\n\n" + tests + "\n", encoding="utf-8")
        rc, out, err = _run([sys.executable, "-s", "-B", "main.py"], str(work), timeout,
                            _env(str(site)), mem_mb << 20 if mem_mb else None)
    if rc is None:
        return _result(False, "test", out, f"timeout after {timeout}s", start)
    if rc != 0:
        return _result(False, "test", out, err.strip().splitlines()[-1] if err.strip() else f"exit {rc}", start,
                       returncode=rc)
    return _result(True, "test", out, "", start, returncode=0)


# ----------------------------------------------------------------------------- SystemVerilog

_FAIL = re.compile(r"\b(fail(ed|ure)?|error)\b|mismatches?:\s*[1-9]|\bTEST FAILED\b", re.I)
_PASS = re.compile(r"mismatches:\s*0\s+in|\bpass(ed)?\b|all tests passed|no error", re.I)


def _top_module(tb: str, default: str = "tb") -> str:
    names = re.findall(r"^\s*module\s+([A-Za-z_]\w*)", tb, re.M)
    return default if default in names else (names[-1] if names else default)


def run_sv(code: str, tests: str = "", timeout: float = 60.0, mem_mb: int | None = None,
           top: str | None = None, pass_regex: str | None = None) -> dict[str, Any]:
    """pyslang parse -> elaboration -> (if a testbench and a simulator exist) simulation.

    ``tests`` is a self-checking testbench source (may also contain a reference model). Pass
    means the simulator exited 0, printed no failure marker and printed a pass marker
    (or matched ``pass_regex``).
    """
    from .judge import validate_sv

    start = time.monotonic()
    v = validate_sv(code)
    if not v["syntax"]["passed"]:
        msgs = [d["message"] for d in v["syntax"]["diagnostics"] if d["severity"].lower().endswith("error")]
        return _result(False, "parse", error="\n".join(msgs) or "syntax error", start=start)
    if not v["elaboration"]["passed"]:
        msgs = [d["message"] for d in v["elaboration"]["diagnostics"] if d["severity"].lower().endswith("error")]
        return _result(False, "elab", error="\n".join(msgs) or "elaboration error", start=start)
    tools = available_tools()
    if not tests.strip():
        return _result(True, "elab", start=start, simulator=None)
    sim = "iverilog" if tools["iverilog"] and tools["vvp"] else "verilator" if tools["verilator"] else None
    if sim is None:
        return _result(True, "elab", error="no simulator installed; testbench not run", start=start, simulator=None)
    top = top or _top_module(tests)
    env = _env()
    with tempfile.TemporaryDirectory(prefix="flyslop-sv-") as tmp:
        (Path(tmp) / "dut.sv").write_text(code, encoding="utf-8")
        (Path(tmp) / "tb.sv").write_text(tests, encoding="utf-8")
        if sim == "iverilog":
            rc, out, err = _run([tools["iverilog"], "-g2012", "-s", top, "-o", "sim.vvp", "dut.sv", "tb.sv"],
                                tmp, timeout, env, mem_mb and mem_mb << 20)
            if rc != 0:
                return _result(False, "test", out, "compile: " + (err or "timeout"), start, simulator=sim)
            rc, out, err = _run([tools["vvp"], "sim.vvp"], tmp, max(1.0, timeout - (time.monotonic() - start)),
                                env, mem_mb and mem_mb << 20)
        else:
            rc, out, err = _run([tools["verilator"], "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                                 "--top-module", top, "-Mdir", "obj", "dut.sv", "tb.sv"],
                                tmp, timeout, env, mem_mb and mem_mb << 20)
            if rc != 0:
                return _result(False, "test", out, "compile: " + (err[-800:] or "timeout"), start, simulator=sim)
            rc, out, err = _run([str(Path(tmp) / "obj" / f"V{top}")], tmp,
                                max(1.0, timeout - (time.monotonic() - start)), env, mem_mb and mem_mb << 20)
    if rc is None:
        return _result(False, "test", out, f"timeout after {timeout}s", start, simulator=sim)
    passed = re.search(pass_regex, out) if pass_regex else (not _FAIL.search(out) and _PASS.search(out))
    ok = rc == 0 and bool(passed)
    return _result(ok, "test", out, "" if ok else (err.strip() or "testbench reported failure or no pass marker"),
                   start, simulator=sim, returncode=rc)


def judge(lang: str, code: str, tests: str = "", **kw: Any) -> dict[str, Any]:
    """Dispatch on ``lang`` ('py' or 'sv'), matching the dataset record format."""
    if lang == "py":
        return run_python(code, tests, **kw)
    if lang == "sv":
        return run_sv(code, tests, **kw)
    raise ValueError(f"unknown lang {lang!r}")


if __name__ == "__main__":
    import json
    print(json.dumps(available_tools(), indent=1))
