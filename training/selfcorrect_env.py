"""Self-correction loop env (PLAN.md stage 9, P7): write -> run -> observe -> edit -> run.

State given to the policy = tokenised ``#TASK prompt / #BUF buffer / #OUT last test output`` (buffer left-truncated,
output tail-truncated). One *round* = one ``policy.act(context_ids)`` call. Its tokens are code tokens or edit
tokens applied through :class:`backend.editor.Editor` (``keyplan.apply_tokens``); ``<RUN>`` runs ``exec_judge`` on
the buffer and ends the round, ``<EOS>`` ends the episode, tokens after the first ``<RUN>``/``<EOS>`` are ignored.

``max_rounds`` = number of *repair* rounds allowed after the first attempt (so at most ``max_rounds + 1`` act calls).
Reward = 1 if the final buffer passes its tests (judged on the final buffer even if the policy never ran it), else 0.
``repair_rounds`` = number of runs after the first run. Termination is guaranteed by: the round limit, a per-run
timeout (``exec_judge`` kills the process group: infinite Python loops, Verilator hangs), a per-round token cap, a
buffer size cap and a whole-episode wall-clock budget.

Policies are duck-typed: ``act(context_ids) -> token ids`` (ints or strings naming tokens). A policy that also has
``act_env(env, context_ids)`` is called with the env instead (used by the oracle policies, which peek at the buffer).
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Iterable, Sequence

from backend import exec_judge
from backend.editor import Editor
from backend.keyplan import apply_tokens, oracle_edits
from training.tokenizer import BOS, EOS, PAD, RUN, BPETokenizer

_RUN_NAMES = ("<RUN>", RUN)
_EOS_NAMES = ("<EOS>", EOS)


class JudgeCache:
    """Thread-safe memo of judge results keyed by (lang, code, tests, timeout)."""

    def __init__(self):
        self._d: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __call__(self, lang: str, code: str, tests: str, timeout: float) -> dict:
        key = hashlib.sha256(f"{lang}\0{code}\0{tests}\0{timeout}".encode()).hexdigest()
        with self._lock:
            if key in self._d:
                self.hits += 1
                return self._d[key]
        res = exec_judge.judge(lang, code, tests, timeout=timeout)
        with self._lock:
            self.misses += 1
            self._d[key] = res
        return res


def _is(tok, names) -> bool:
    return (tok in names) if not isinstance(tok, bool) else False


class SelfCorrectEnv:
    def __init__(self, tokenizer: BPETokenizer | None = None, max_rounds: int = 3, max_tokens_per_round: int = 512,
                 run_timeout: float = 20.0, episode_timeout: float = 300.0, max_buffer_chars: int = 4000,
                 max_obs_chars: int = 240, max_ctx: int = 1024, judge=None):
        self.tok = tokenizer or BPETokenizer()
        self.max_rounds, self.max_tokens_per_round = max_rounds, max_tokens_per_round
        self.run_timeout, self.episode_timeout = run_timeout, episode_timeout
        self.max_buffer_chars, self.max_obs_chars, self.max_ctx = max_buffer_chars, max_obs_chars, max_ctx
        self.judge = judge or JudgeCache()

    # ------------------------------------------------------------------ state
    def reset(self, task: dict, start_buffer: str | None = None, cursor: int | None = None) -> list[int]:
        self.task = task
        self.editor = Editor(start_buffer or "", cursor)
        self.rounds = 0
        self.runs: list[dict] = []
        self.last: dict | None = None
        self.last_hash: str | None = None      # buffer hash of the last run
        self.done, self.reason, self.reward = False, "", 0.0
        self.t0 = time.monotonic()
        self.tokens_used = 0
        self.keystrokes = 0
        if start_buffer is not None:           # a supplied buffer is run once so the policy sees its failure
            self._run()
            if self.last["ok"]:                # already correct: nothing to repair, episode ends (terminated)
                self.done, self.reason = True, "pass"
                self._finish()
        return self.context()

    def _run(self) -> dict:
        code = self.editor.text
        res = self.judge(self.task["lang"], code, self.task["tests"], self.run_timeout)
        self.last, self.last_hash = res, hashlib.sha256(code.encode()).hexdigest()
        self.runs.append({"ok": bool(res["ok"]), "stage": res["stage"], "duration": res.get("duration", 0.0)})
        return res

    def observation_text(self) -> str:
        if self.last is None:
            return "no run yet"
        r = self.last
        msg = f"[{r['stage']}] {'PASS' if r['ok'] else 'FAIL'} {r.get('error', '')} {r.get('stdout_tail', '')}"
        msg = " ".join(msg.split())
        return msg[-self.max_obs_chars:]

    def context(self) -> list[int]:
        enc = self.tok.encode
        head = [BOS] + enc("#TASK ") + enc(self.task["prompt"]) + enc("\n#BUF\n")
        tail = enc("\n#OUT\n") + enc(self.observation_text())
        buf = enc(self.editor.text)
        room = max(0, self.max_ctx - len(head) - len(tail))
        return head + (buf[-room:] if room else []) + tail

    # ------------------------------------------------------------------ dynamics
    def step(self, tokens: Iterable) -> tuple[list[int], float, bool, dict]:
        assert not self.done, "episode finished"
        toks = list(tokens)[: self.max_tokens_per_round]
        toks = [t for t in toks if not _is(t, (PAD, "<PAD>"))]
        self.rounds += 1
        self.tokens_used += len(toks)
        cut = next((i for i, t in enumerate(toks) if _is(t, _RUN_NAMES) or _is(t, _EOS_NAMES)), None)
        end = toks[cut] if cut is not None else None
        edits = toks if cut is None else toks[:cut]
        for t in edits:                       # one token at a time so the buffer cap bounds runaway generation
            if len(self.editor.text) > self.max_buffer_chars:
                break
            try:
                apply_tokens(self.editor, [t], self.tok)
                self.keystrokes += 1
            except Exception:                 # unknown / invalid token: skip
                continue
        if len(self.editor.text) > self.max_buffer_chars:
            self.editor.text = self.editor.text[: self.max_buffer_chars]
            self.editor.cursor = min(self.editor.cursor, len(self.editor.text))
            self.editor.anchor = min(self.editor.anchor, len(self.editor.text))
        if _is(end, _RUN_NAMES):
            self._run()
            if self.last["ok"]:
                self.done, self.reason = True, "pass"
        elif end is not None:
            self.done, self.reason = True, "eos"
        if not self.done and self.rounds >= self.max_rounds + 1:
            self.done, self.reason = True, "max_rounds"
        if not self.done and time.monotonic() - self.t0 > self.episode_timeout:
            self.done, self.reason = True, "budget"
        if self.done:
            self._finish()
        return self.context(), self.reward, self.done, self.info()

    def _finish(self) -> None:
        h = hashlib.sha256(self.editor.text.encode()).hexdigest()
        if self.last is None or self.last_hash != h:     # final buffer never run: judge it for the reward
            self._run()
        self.reward = 1.0 if self.last["ok"] else 0.0

    def info(self) -> dict[str, Any]:
        return {"reason": self.reason, "rounds": self.rounds, "n_runs": len(self.runs),
                "repair_rounds": max(0, len(self.runs) - 1), "first_ok": self.runs[0]["ok"] if self.runs else False,
                "ok": self.reward > 0, "tokens": self.tokens_used, "keystrokes": self.keystrokes,
                "elapsed": round(time.monotonic() - self.t0, 3), "buffer": self.editor.text}

    # ------------------------------------------------------------------ driver
    def run_episode(self, policy, task: dict, start_buffer: str | None = None, cursor: int | None = None) -> dict:
        ctx = self.reset(task, start_buffer, cursor)
        info = self.info()
        while not self.done:
            toks = policy.act_env(self, ctx) if hasattr(policy, "act_env") else policy.act(ctx)
            ctx, _, _, info = self.step(_as_list(toks))
        info["id"] = task["id"]
        return info


def _as_list(x) -> list:
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


# ---------------------------------------------------------------------- policies
class OracleRepairPolicy:
    """Emit ``oracle_edits(buffer -> reference)`` then ``<RUN>``. From an empty buffer this writes the reference."""

    def act_env(self, env: SelfCorrectEnv, ctx) -> list:
        ed = env.editor
        return oracle_edits(ed.text, ed.cursor, ed.selection, env.task["code"]) + ["<RUN>"]

    def act(self, ctx):
        raise RuntimeError("OracleRepairPolicy needs the env (use env.run_episode)")


class NullPolicy:
    """Emits <EOS> immediately: never writes anything."""

    def act(self, ctx):
        return [EOS]


class ThinkerPolicy:
    """Adapter for a ``backend.connectome.thinker`` model: greedy ``generate`` on the context ids."""

    def __init__(self, model, max_new: int = 256):
        self.model, self.max_new = model, max_new

    def act(self, ctx):
        out = self.model.generate([list(ctx)], max_new=self.max_new, eos=EOS)
        row = out[0].tolist() if hasattr(out[0], "tolist") else list(out[0])
        return [t for t in row if t != PAD]


def corrupted_start(task: dict, seed: int, n: int = 1):
    from backend.error_inject import corrupt_buffer
    return corrupt_buffer(task["code"], seed=seed, n=n)
