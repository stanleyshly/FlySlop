"""Physical-eval harness: symbolic action tokens -> key commands -> the fly types them in MuJoCo.

    uv run --extra training python -m training.physical_eval --text "def f(x):\\n    return x" \\
        --typist expert --action-mode mn [--tokens-file toks.json] [--replay-out out.json]

``type_tokens`` expands tokens (or plain text) with ``backend.keyplan``, compiles the key commands into the
env's key-command queue (``FlyTypingEnv.reset(options={"keys": ...})``) and steps the physical fly. Text comes
only from foot contacts: every ``key`` event is checked by ``embodiment.replay_text`` (contact -> character).

Chords (``chords=True``, the default for the scripted expert) use **held modifiers**: ``shift_mode="held"`` and a
two-foreleg chord expert (``FlyTypingEnv._chord_expert``): one foreleg presses and holds Shift or Fn while the other
strikes the key, then both lift; a ``<SEL_START> ... <SEL_END>`` selection keeps Shift down across the arrows when the
next press is still in reach. Plain text without chords keeps the legacy one-shot Shift latch (``shift_mode="auto"``),
which is what the PPO policies were trained with, so their behaviour is unchanged.

Geometry limit: the forelegs are about 5 key pitches apart and span at most about 11, but Fn sits at the far left of
the bottom row and Backspace, ArrowLeft and ArrowRight at the far right (13 to 14 pitches away), so forward delete, Home
and End cannot be held as chords. ``fn_mode="auto"`` then falls back to ``fn_mode="latch"``: a tap on Fn arms the
next press (sticky-keys style, ``PhysicsParams.fn_mode``), and the run reports ``fn_latch_used``. ``fn_mode="held"``
raises/reports :class:`Unsupported` with the reach numbers instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.editor import Editor
from backend.embodiment import replay_text, text_hash
from backend.fly_env import UnreachableChord
from backend.keyplan import SHIFT, KeyCommand, apply_commands, expand_tokens, oracle_edits, text_to_keys
from backend.memory_budget import start_watchdog

SHIFT_KEYS = ("ShiftLeft", "ShiftRight")


class Unsupported(ValueError):
    """The physical typists cannot realise this command (see module docstring)."""


def compile_commands(commands, chords: bool = False) -> tuple[list[tuple], list[int]]:
    """Key commands -> the env queue plus the command index of each queue item.

    ``chords=False`` (legacy): items are ``(key_id, shifted)``, a held-Shift region becomes ``shifted`` on every press
    (the latch re-taps it) and an Fn chord raises :class:`Unsupported`. ``chords=True``: items are
    ``(key_id, shifted, extra)`` with ``extra`` = ``("Fn",)`` for an Fn chord; the env presses them as held chords.
    """
    keys, owner, held = [], [], False
    for i, cmd in enumerate(commands):
        if cmd.key in SHIFT_KEYS and cmd.hold:
            held = True
        elif cmd.release:
            held = False
        else:
            fn = "Fn" in cmd.mods
            if fn and not chords:
                raise Unsupported(f"Fn chord {cmd.key}+Fn: two forelegs would have to hold Fn and the key at once")
            shifted = held or any(m in SHIFT_KEYS for m in cmd.mods)
            keys.append((cmd.key, shifted, ("Fn",) if fn else ()) if chords else (cmd.key, shifted))
            owner.append(i)
    return keys, owner


def _commands(source) -> list[KeyCommand]:
    if isinstance(source, str):
        return text_to_keys(source)
    items = list(source)
    if items and all(isinstance(c, KeyCommand) for c in items):
        return items
    return expand_tokens(items)


def _state(editor: Editor) -> dict:
    return {"text": editor.text, "cursor": editor.cursor, "selection": editor.selection}


def symbolic(commands) -> dict:
    ed = Editor()
    apply_commands(ed, commands)
    return _state(ed)


def _load_typist(typist: str, action_mode: str, record: bool, physics: dict | None = None):
    from backend.fly_env import FlyTypingEnv
    if typist == "expert":
        return FlyTypingEnv(targets=("a",), action_mode=action_mode, terminate_on_error=False, record=record,
                            physics=physics), None
    from backend.policy_replay import _run_config
    from training.common import make_env
    from training.train_ppo import load_ppo
    path = Path(typist)
    config = _run_config(path)
    model = load_ppo(path, device="cpu")
    env = make_env(config, ["a"], record=record, terminate_on_error=False)
    return env, model


def _fn_free_plan(state: dict, target: str) -> list[KeyCommand]:
    """Fallback plan without Fn chords: walk to the end, Backspace everything, retype."""
    n = len(state["text"])
    return (expand_tokens(["<RIGHT>"] * n) + expand_tokens(["<BS>"] * n) + text_to_keys(target))


def _replan(state: dict, target: str, env=None, chords: bool = False) -> list[KeyCommand]:
    toks = oracle_edits(state["text"], state["cursor"], state["selection"], target)
    cmds = expand_tokens(toks)
    try:
        keys, _ = compile_commands(cmds, chords)
        if chords and env is not None:
            for item in keys:
                env.check_chord(env_item(item))
        return cmds
    except (Unsupported, UnreachableChord):
        return _fn_free_plan(state, target)


def env_item(item: tuple) -> tuple:
    """Queue item as the env stores it: ``(key_id, shifted, extra)``."""
    return item if len(item) == 3 else (*item, ())


def _press_units(keys: list[tuple]) -> int:
    """Press-equivalents the tick budget must cover: a chord costs about three (walk, hold, strike)."""
    return len(keys) + sum(bool(k[1]) for k in keys) + 2 * sum(len(k) > 2 and "Fn" in k[2] for k in keys)


def type_tokens(source, typist: str = "expert", action_mode: str = "mn", seed: int = 0, max_ticks: int | None = None,
                recover: bool = True, max_replans: int = 6, record: bool = False, env=None, model=None,
                noise: float = 0.0, chords: bool | None = None, shift_mode: str = "auto", fn_mode: str = "auto") -> dict:
    """Type ``source`` (text, tokens, or KeyCommands) on the physical fly.

    Returns a dict: ``text`` (typed buffer), ``events`` (editor-annotated env events), ``success_rate`` (commands
    typed on the first try / commands), ``ticks``, ``exact`` (typed text/cursor/selection == symbolic
    ``apply_tokens`` result), ``text_match``, ``contact_verified`` (+ ``contact_text`` / ``contact_error``),
    ``slips`` (each with tick, expected key, typed key, buffer), ``replans``, ``complete``, wall-clock times.

    ``chords`` (default: only for the scripted expert) enables held-modifier chords. ``shift_mode``: ``"auto"``
    (held when the commands contain Fn chords or a Shift selection, else the legacy latch), ``"held"`` or ``"latch"``.
    ``fn_mode``: ``"auto"`` (held, falling back to the Fn latch when a chord is out of the forelegs' reach), ``"held"``
    or ``"latch"``. The result reports the modes used and ``unreachable`` (why a held chord could not be planned).
    """
    t0 = time.time()
    commands = _commands(source)
    want = symbolic(commands)
    if chords is None:
        chords = typist == "expert" and (env is None or env.sim.params.shift_mode == "held" or shift_mode != "latch")
    shift_req = shift_mode
    has_fn = any("Fn" in c.mods for c in commands)
    has_sel = any(c.key in SHIFT_KEYS and c.hold for c in commands)
    if env is not None:
        shift_mode, fn_mode = env.sim.params.shift_mode, env.sim.params.fn_mode
    else:
        if shift_mode == "auto":
            shift_mode = "held" if (chords and (has_fn or has_sel)) else "latch"
        if not chords:
            shift_mode = "latch"
    out = {"n_commands": len(commands), "expected": want, "typist": typist, "action_mode": action_mode,
           "seed": seed, "slips": [], "replans": 0, "unsupported": None, "unreachable": None, "shift_mode": shift_mode,
           "fn_mode": fn_mode if has_fn else None, "fn_latch_used": False, "shift_latch_used": False}

    def fail(reason: str) -> dict:
        return {**out, "text": "", "events": [], "success_rate": 0.0, "ticks": 0, "exact": False, "text_match": False,
                "contact_verified": False, "complete": False, "wall_s": time.time() - t0}

    try:
        keys, owner = compile_commands(commands, chords)
    except Unsupported as exc:
        out["unsupported"] = str(exc)
        if not recover:
            return fail(str(exc))
        n_fn = sum("Fn" in c.mods for c in commands)
        out["unsupported_fraction"] = n_fn / max(1, len(commands))
        commands = _replan({"text": "", "cursor": 0, "selection": None}, want["text"])   # plan from empty
        keys, owner = compile_commands(commands)
    if not keys:
        return {**out, "text": "", "events": [], "success_rate": 1.0, "ticks": 0, "exact": want["text"] == "",
                "text_match": want["text"] == "", "contact_verified": True, "complete": True, "wall_s": time.time() - t0}
    own_env = env is None
    # Ladder of physical configurations, most faithful first: held modifiers, then a sticky Fn for chords the forelegs
    # cannot span, then (only for shift_mode="auto") the legacy Shift latch for Shift chords out of reach (e.g. "_").
    configs = [(shift_mode, "latch" if fn_mode == "latch" else "held")]
    if own_env and typist == "expert":
        if fn_mode == "auto" and has_fn:
            configs.append((shift_mode, "latch"))
        if shift_req == "auto" and shift_mode == "held":
            configs.append(("latch", configs[-1][1] if has_fn else "held"))
            if has_fn and configs[-1][1] == "held" and fn_mode == "auto":
                configs.append(("latch", "latch"))
    env_cache, base_env, base_model = {}, env, model
    ladder = iter(configs)
    while True:
        cfg = next(ladder)
        if own_env:
            if cfg not in env_cache:
                env_cache[cfg] = _load_typist(typist, action_mode, record, {"shift_mode": cfg[0], "fn_mode": cfg[1]})
            env, model = env_cache[cfg]
        budget = max_ticks or env.base_steps + env.steps_per_char * _press_units(keys)
        try:
            obs, _ = env.reset(seed=seed, options={"keys": keys, "max_steps": budget})
            break
        except UnreachableChord as exc:
            out["unreachable"] = out["unreachable"] or str(exc)
            if cfg == configs[-1]:
                env = None
                break
    if env is not None and own_env:
        out["shift_mode"] = cfg[0]
        out["fn_mode"] = cfg[1] if has_fn else None
        out["fn_latch_used"] = bool(has_fn and cfg[1] == "latch" and configs[0][1] == "held")
        out["shift_latch_used"] = cfg[0] == "latch" and configs[0][0] == "held"
    if env is None:
        if not recover:
            return fail(out["unreachable"])
        out["unsupported"] = out["unreachable"]
        n_bad = sum(1 for k in keys if len(k) > 2 and k[2])
        out["unsupported_fraction"] = n_bad / max(1, len(commands))
        commands = _replan({"text": "", "cursor": 0, "selection": None}, want["text"])
        keys, owner = compile_commands(commands)
        cfg = configs[-1]
        if own_env:
            env, model = env_cache.get(cfg) or _load_typist(typist, action_mode, record,
                                                            {"shift_mode": cfg[0], "fn_mode": cfg[1]})
        else:
            env, model = base_env, base_model
        budget = max_ticks or env.base_steps + env.steps_per_char * _press_units(keys)
        try:
            obs, _ = env.reset(seed=seed, options={"keys": keys, "max_steps": budget})
        except UnreachableChord as exc:
            out["unreachable"] = str(exc)
            return fail(str(exc))
    rng = np.random.default_rng(seed)
    bad = set()          # original command indices that slipped
    n_first = len(commands)
    done, replans, complete = False, 0, False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.expert_action()
        if noise:                                       # motor noise on every 3rd tick to provoke slips
            if env.tick % 3 == 0:
                action = np.clip(action + rng.normal(0, noise, action.shape), -1, 1).astype(np.float32)
        pending = env.qi
        obs, _r, terminated, truncated, info = env.step(action)
        if info["error"]:
            slip = [e for e in info["events"] if e["type"] == "key"][-1]
            out["slips"].append({"tick": env.tick, "expected": env.queue[pending][0] if pending < len(env.queue) else None,
                                 "typed": slip["key_id"], "char": slip["char"], "text": env.editor.text})
            if replans == 0 and pending < len(owner):
                bad.add(owner[pending])
            if recover and replans < max_replans:
                replans += 1
                cmds = _replan(_state(env.editor), want["text"], env, chords)
                keys2, owner2 = compile_commands(cmds, chords)
                if not keys2:
                    complete = True
                    break
                env.set_keys(keys2)
                owner = [-1] * len(owner2)
                env.max_steps = max(env.max_steps, env.tick + env.base_steps + env.steps_per_char * _press_units(keys2) * 2)
                continue
            done = True
        done = done or terminated or truncated
        complete = terminated and not info["error"]
    events = list(env.events)
    result = _state(env.editor)
    try:
        contact_text, contact_error = replay_text(events), None
    except ValueError as exc:                          # rejected: contactless or inconsistent event
        contact_text, contact_error = None, str(exc)
    verified = contact_error is None and contact_text == result["text"]
    ticks = env.tick
    unfinished = env.qi < len(env.queue) and replans == 0
    if unfinished:
        bad.update(owner[env.qi:])
    out.update({
        "text": result["text"], "state": result, "events": events, "ticks": ticks, "complete": bool(complete),
        "success_rate": (1.0 - len(bad) / max(1, n_first)) * (1.0 - out.get("unsupported_fraction", 0.0)), "replans": replans,
        "text_match": result["text"] == want["text"], "exact": result == want,
        "contact_verified": verified, "contact_text": contact_text, "contact_error": contact_error,
        "ticks_per_command": ticks / max(1, n_first), "wall_s": time.time() - t0,
        "wall_s_per_command": (time.time() - t0) / max(1, n_first), "env": env,
    })
    return out


def verify_events(events: list[dict], expected_text: str) -> tuple[bool, str | None]:
    """Contact-to-character verification of an event list against the text it should have produced."""
    try:
        return replay_text(events) == expected_text, None
    except (ValueError, KeyError) as exc:
        return False, str(exc)


def write_replay(result: dict, path, target_id: str = "physical_eval") -> dict:
    """Viewer-loadable replay JSON (the ``backend/server.py`` replay format) from a ``record=True`` run."""
    from backend.judge import validate_sv
    from backend.keyboard import layout_payload
    env = result["env"]
    if not env.record:
        raise ValueError("run type_tokens(record=True) to write a replay")
    replay = env.replay(target_id, "physical_eval_key_commands")
    replay["target"] = result["expected"]["text"]
    replay["layout"] = layout_payload()
    replay["metadata"]["typist"] = result["typist"]
    replay["metadata"]["action_mode"] = result["action_mode"]
    replay["metadata"]["slips"] = len(result["slips"])
    if replay_text(replay["events"]) != replay["final_text"]:
        raise RuntimeError("Replay failed integrity check")
    validation = validate_sv(replay["final_text"])
    validation.update({"target_id": target_id,
                       "exact": {"passed": replay["final_text"] == replay["target"],
                                 "expected_length": len(replay["target"]), "actual_length": len(replay["final_text"])},
                       "text": {"expected": replay["target"], "actual": replay["final_text"]}})
    replay["validation"] = validation
    Path(path).write_text(json.dumps(replay, separators=(",", ":")))
    return replay


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to type (\\n and \\t escapes are expanded)")
    src.add_argument("--tokens-file", help="JSON list of action tokens (strings or ids)")
    parser.add_argument("--typist", default="expert", help='"expert" or a PPO checkpoint .zip')
    parser.add_argument("--action-mode", default="mn", choices=["claw", "mn"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--no-recover", action="store_true")
    parser.add_argument("--shift-mode", default="auto", choices=["auto", "held", "latch"])
    parser.add_argument("--fn-mode", default="auto", choices=["auto", "held", "latch"])
    parser.add_argument("--replay-out", help="write a viewer replay JSON here")
    args = parser.parse_args(argv)
    start_watchdog()
    if args.text is not None:
        source = args.text.encode().decode("unicode_escape")
    else:
        source = json.loads(Path(args.tokens_file).read_text())
    res = type_tokens(source, args.typist, args.action_mode, args.seed, args.max_ticks, recover=not args.no_recover,
                      record=bool(args.replay_out), shift_mode=args.shift_mode, fn_mode=args.fn_mode)
    if args.replay_out:
        write_replay(res, args.replay_out)
    summary = {k: res[k] for k in ("n_commands", "success_rate", "ticks", "exact", "text_match", "contact_verified",
                                   "replans", "complete", "ticks_per_command", "wall_s", "wall_s_per_command",
                                   "unsupported", "unreachable", "shift_mode", "fn_mode", "fn_latch_used", "shift_latch_used") if k in res}
    summary["slips"] = len(res["slips"])
    summary["text_hash"] = text_hash(res.get("text", ""))
    print(json.dumps(summary, indent=1))
    print("typed:", repr(res.get("text")))
    return 0 if res.get("text_match") else 1


if __name__ == "__main__":
    raise SystemExit(main())
