"""Gymnasium environment: the NeuroMechFly body types on the MuJoCo keyboard.

Control hierarchy (PLAN §5.1): the policy commands (1) a thorax walking
velocity, which an online gait turns into stance/swing targets for the four
middle and hind legs, and (2) absolute claw targets for both forelegs (RF and
LF), the typing legs, each inside a reach box around its press anchor. IK turns claw
targets into 42 joint targets; MuJoCo position actuators drive the joints and
claw contacts push spring-loaded keys. Characters come only from
:class:`backend.physics.PhysicalKeyboard` (key travel plus claw contact).

The observation is state assisted: it includes the offset to the next
physical key (Shift before a shifted character) from each foreleg, and which
foreleg's side of the body the key is on. A key actuated by a walking
leg counts like any other character, so bad footholds are wrong keypresses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # optional extra
    raise ImportError("FlyTypingEnv requires the optional 'gymnasium' dependency") from exc

from .editor import Editor
from .embodiment import (BOUNDS, FACE_YAW, FLY_SCALE, GAIT_HALF, HOVER, MAX_SPEED, MM, PRESS_ANCHOR, REACH, STAND_HEIGHT,
                         SWING_HEIGHT,
                         Planner, annotate_key_event, ground, smoothstep, synthetic_activity, text_hash, to_layout)
from .keyboard import BY_ID, CHAR_TO_KEY, KEY_TRAVEL, LAYOUT, hit_test
from .physics import LEGS, FlySim, LegIK, PhysicsParams, key_world_center, support_height

SHIFT_IDS = ("ShiftLeft", "ShiftRight")
CHORD_TOL = 0.08         # max IK shortfall (key pitches) for a press point to count as reachable by a foreleg
CHORD_BOUNDS = ((-6.4, 6.4), BOUNDS[1])   # thorax XY limits while chords are queued: a chord near an edge of the
                                          # keyboard needs the thorax further out than the default limits allow
CHORD_GRID = 0.25        # thorax XY search resolution for chord standing spots (key pitches)

RF, LF = LEGS.index("RF"), LEGS.index("LF")
TYPERS = ("RF", "LF")                            # action blocks [2:5] and [5:8]
TYPER_INDEX = [RF, LF]
GROUPS = ({"RM", "LH"}, {"LM", "RH"})           # diagonal pairs of the four walking legs
# Half-widths (world X, Y) of each foreleg reach box, in key pitches. Facing the
# screen, world X is the body's lateral axis and world Y its forward axis.
REACH_XY = np.array([REACH[1], REACH[0]]) * MM

# --- action_mode="mn": 16-D action = (vx, vy) thorax velocity + 7 joint targets for RF then LF, in
# ``default_body().leg_dofs`` order. A joint target is ``rest + a * JOINT_SCALE`` (a in [-1, 1]). The scale is
# ~1.25x the largest excursion of the scripted expert's IK solution per DOF (measured on 600 expert steps).
JOINT_DOFS = ("Coxa_yaw", "Coxa", "Coxa_roll", "Femur", "Femur_roll", "Tibia", "Tarsus1")
JOINT_SCALE = np.array([0.35, 0.5, 0.45, 0.7, 0.35, 1.1, 0.35])
JOINT_NAMES = tuple(f"joint_{leg}{dof}" for leg in TYPERS for dof in JOINT_DOFS)   # RF block, then LF block
KEY_IDS = tuple(key.id for key in LAYOUT)
KEY_LABELS = tuple(f"key_{k}" for k in KEY_IDS)
EDITOR_LABELS = ("ed_cursor", "ed_tail1", "ed_tail2", "ed_tail3")
# Observation slots that reveal where the next key is (the "state assisted" offsets); zeroed by state_assist=False.
ASSIST_LABELS = ("anchor_dx", "anchor_dy", "tip_dx", "tip_dy", "lf_anchor_dx", "lf_anchor_dy", "lf_tip_dx",
                 "lf_tip_dy", "key_side")


class UnreachableChord(ValueError):
    """No thorax pose lets the two forelegs hold the modifier and press the key at the same time."""


def normalize_key_items(keys) -> list[tuple[str, bool, tuple[str, ...]]]:
    """Key-command queue items -> ``(key_id, shifted, extra)``.

    Accepted: legacy ``(key_id, shifted)`` tuples, ``(key_id, mods)`` with ``mods`` a sequence of modifier ids, and
    C2 commands (dicts or objects with ``key``, ``mods``, ``hold``, ``release``). ``hold``/``release`` of a Shift key
    are folded into the items between them. ``extra`` holds ``"Fn"`` and/or an explicit Shift key id.
    """
    out, sticky = [], None
    for item in keys:
        hold = release = False
        shifted = False
        if isinstance(item, dict) or hasattr(item, "mods"):
            get = item.get if isinstance(item, dict) else (lambda name, default=None: getattr(item, name, default))
            key, mods = str(get("key")), tuple(get("mods", ()) or ())
            hold, release = bool(get("hold", False)), bool(get("release", False))
        else:
            item = tuple(item)
            key, second = str(item[0]), (item[1] if len(item) > 1 else False)
            if isinstance(second, (bool, int, np.bool_)):
                shifted, mods = bool(second), (tuple(item[2]) if len(item) > 2 else ())
            else:
                mods = tuple(second)
        if key not in BY_ID:
            raise ValueError("unknown key id in queue")
        bad = [m for m in mods if m not in SHIFT_IDS and m != "Fn"]
        if bad:
            raise ValueError(f"modifiers {bad} are not physically used (only Shift and Fn, contract C2)")
        if release:
            sticky = None
            continue
        if hold:
            if key not in SHIFT_IDS:
                raise ValueError("only a Shift key can be held")
            sticky = key
            continue
        shift_id = next((m for m in mods if m in SHIFT_IDS), sticky)
        extra = (("Fn",) if "Fn" in mods else ()) + ((shift_id,) if shift_id else ())
        out.append((key, shifted or shift_id is not None, extra))
    return out


_REACH_TABLES: dict[str, np.ndarray] = {}   # foreleg -> IK shortfall on a 0.25 grid of press points around its anchor


def obs_labels(key_obs: bool = False, editor_obs: bool = False) -> tuple[str, ...]:
    return (*FlyTypingEnv.OBS_LABELS, *(KEY_LABELS if key_obs else ()), *(EDITOR_LABELS if editor_obs else ()))


def physical_key(char: str, shift_latched: bool) -> str:
    """Next key to press for ``char``: the nearer Shift first when needed."""
    key_id, needs_shift = CHAR_TO_KEY[char]
    return physical_key_id(key_id, needs_shift, shift_latched)


def physical_key_id(key_id: str, needs_shift: bool, shift_latched: bool) -> str:
    """``physical_key`` for a key id plus a shift flag (used by the key-command queue mode)."""
    if not needs_shift or shift_latched:
        return key_id
    key, left, right = BY_ID[key_id], BY_ID["ShiftLeft"], BY_ID["ShiftRight"]
    return "ShiftLeft" if abs(key.x - (left.x + left.width)) <= abs(right.x - key.x) else "ShiftRight"


def press_point(key_id: str, near_x: float) -> np.ndarray:
    """World XY to press: key centre, or the nearest inset point along a long key."""
    return Planner.press_point(BY_ID[key_id], np.array([near_x, 0.0]))


class OnlineGait:
    """Diagonal-pair gait of the four walking legs that follows a commanded thorax velocity.

    Every ``GAIT_HALF`` ticks one group swings toward nominal footholds at the
    thorax pose predicted one half-cycle ahead. Groups swing only while the
    body moves or a foot is off its nominal foothold.
    """

    def __init__(self, planner: Planner, feet: np.ndarray):
        self.planner = planner
        self.feet = feet.copy()
        self.swings: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        self.group = 0
        self.k = 0

    def step(self, body: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        if self.k % GAIT_HALF == 0:
            ahead = body.copy()
            ahead[:2] += velocity * (GAIT_HALF * 1.5)
            for leg in GROUPS[self.group]:
                i = LEGS.index(leg)
                target = self.planner.nominal_world(ahead, leg)
                if i not in self.swings and np.linalg.norm(target[:2] - self.feet[i, :2]) > 0.04:
                    self.swings[i] = (self.feet[i].copy(), target, self.k)
            self.group = 1 - self.group
        for i, (lift, land, t0) in list(self.swings.items()):
            s = (self.k - t0 + 1) / GAIT_HALF
            xy = lift[:2] + (land[:2] - lift[:2]) * smoothstep(s)
            # sin^2 arc: zero vertical speed at lift-off and touchdown (soft landings).
            z = lift[2] + (land[2] - lift[2]) * smoothstep(s) + SWING_HEIGHT * math.sin(math.pi * min(1.0, s)) ** 2
            self.feet[i] = [xy[0], xy[1], z]
            if s >= 1:
                self.feet[i] = land
                del self.swings[i]
        self.k += 1
        return self.feet


class FlyTypingEnv(gym.Env):
    """Type a target string with the physically simulated fly.

    Action (all in ``[-1, 1]``): ``vx, vy`` thorax velocity (x ``MAX_SPEED``),
    ``rx, ry`` RF claw target in the reach box around the RF anchor, ``rz``
    RF claw height from the bottom of key travel (minus push-through) to hover,
    then ``lx, ly, lz``, the same for LF.
    """

    metadata = {"render_modes": []}
    OBS_LABELS = ("anchor_dx", "anchor_dy", "tip_dx", "tip_dy", "tip_z", "tip_vz", "key_depression", "rf_holding",
                  "shift_needed", "shift_latched", "progress", "time_left", "last_correct",
                  "body_vx", "body_vy", "body_x", "body_y", "prev_vx", "prev_vy", "prev_rx", "prev_ry", "prev_rz",
                  "lf_anchor_dx", "lf_anchor_dy", "lf_tip_dx", "lf_tip_dy", "lf_tip_z", "lf_tip_vz", "lf_holding",
                  "prev_lx", "prev_ly", "prev_lz", "key_side")
    DEFAULT_REWARD = {"correct": 1.0, "wrong": -1.0, "shift": 0.5, "complete": 2.0, "time": -0.002, "shaping": 1.0}

    def __init__(self, targets: tuple[str, ...] = ("fly",), steps_per_char: int = 150, base_steps: int = 100,
                 terminate_on_error: bool = True, gamma: float = 0.99, reward: dict | None = None,
                 physics: dict | None = None, record: bool = False, ik_iterations: int = 3,
                 max_xy_rate: float = 0.25, max_z_rate: float = 0.12, speed_fraction: float = 0.7,
                 action_mode: str = "claw", key_obs: bool | None = None, state_assist: bool = True,
                 editor_obs: bool = False, max_joint_rate: float = 0.8):
        super().__init__()
        if action_mode not in ("claw", "mn"):
            raise ValueError("action_mode must be 'claw' or 'mn'")
        self.action_mode = action_mode
        self.key_obs = (action_mode == "mn") if key_obs is None else bool(key_obs)
        self.state_assist, self.editor_obs = bool(state_assist), bool(editor_obs)
        self.max_joint_rate = float(max_joint_rate)
        self.obs_labels = obs_labels(self.key_obs, self.editor_obs)
        self.action_dim = 8 if action_mode == "claw" else 16
        self.set_targets(targets)
        self.steps_per_char, self.base_steps = int(steps_per_char), int(base_steps)
        self.terminate_on_error = bool(terminate_on_error)
        self.gamma = float(gamma)
        self.reward_weights = {**self.DEFAULT_REWARD, **(reward or {})}
        self.record = bool(record)
        self.ik_iterations = int(ik_iterations)
        self.max_rate = np.array([max_xy_rate, max_xy_rate, max_z_rate])
        self.max_speed = MAX_SPEED * float(speed_fraction)   # physical legs lag faster gaits
        self.sim = FlySim(PhysicsParams(**(physics or {})))
        self.ik = LegIK(self.sim)
        p = self.sim.params
        self.z_low = -KEY_TRAVEL - p.push_through
        if action_mode == "mn":
            walkers = [leg for leg in LEGS if leg not in TYPERS]
            self.ik_walk = LegIK(self.sim, legs=walkers)     # the 4 non-typing legs; typers are policy-driven
            self.ik_typers = LegIK(self.sim, legs=TYPERS)    # expert labels only
            self.joint_rest = np.array([self.ik.rest[i] for i in TYPER_INDEX])   # (2, 7)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(len(self.obs_labels),), dtype=np.float32)
        self.target, self.buffer, self.typed = "", "", ""
        self.queue: list | None = None       # key-command queue mode: [(key_id, shifted, extra), ...] (see reset options["keys"])
        self.qi = 0
        self._chord_cache: dict = {}
        self._chord_plan_cache = None
        self._reach_planner: Planner | None = None
        self._wide = False                   # True while the queue holds chords: the thorax may use CHORD_BOUNDS
        self.editor = Editor()
        self.events: list[dict[str, Any]] = []
        self.rows: list[list[float]] = []

    def set_targets(self, targets) -> None:
        targets = tuple(targets)
        if not targets or any(not t for t in targets):
            raise ValueError("targets must contain non-empty strings")
        bad = sorted({c for t in targets for c in t if c not in CHAR_TO_KEY})
        if bad:
            raise ValueError(f"Unsupported target characters: {bad}")
        self.targets = targets

    # -- geometry helpers ----------------------------------------------
    def anchor_world(self, leg: str = "RF") -> np.ndarray:
        return self.planner.body_point(self.body, PRESS_ANCHOR[leg])

    def typing_leg(self) -> str:
        """The foreleg on the next key's side of the body (LF for keys left of the midline)."""
        key_id = self.next_key()
        if key_id is None:
            return "RF"
        chord = self._current_chord()
        if chord is not None:
            return self._chord_plan(chord)["key_leg"]
        point = press_point(key_id, float(self.body[0]))
        return "LF" if self.planner.to_body(self.body, point)[1] > 0 else "RF"

    def set_keys(self, keys, start: int = 0) -> None:
        """Key-command queue mode: replace the pending presses with ``keys`` = [(key_id, shifted), ...] without
        resetting the physical state or the editor (used to re-plan after a slip)."""
        keys = normalize_key_items(keys)
        if not keys:
            raise ValueError("keys must be non-empty")
        for item in keys:
            self.check_chord(item)
        self.queue, self.qi = keys, int(start)
        self.target, self.buffer = "\0" * len(keys), "\0" * self.qi
        self._chord_plan_cache = None
        self._wide = any(self._hold_mods(item) for item in keys)

    # -- held-modifier chords (contract C2) -------------------------------
    def _current_item(self):
        """``(key_id, shifted, extra)`` of the press the env is waiting for, in queue or text mode."""
        if self.queue is not None:
            return self.queue[self.qi] if self.qi < len(self.queue) else None
        if len(self.buffer) >= len(self.target):
            return None
        key_id, shifted = CHAR_TO_KEY[self.target[len(self.buffer)]]
        return key_id, shifted, ()

    def _hold_mods(self, item) -> list[str]:
        """Modifiers a foreleg must hold down while ``item``'s key is struck: ``"Fn"``, ``"Shift"`` (either Shift key)
        or an explicit Shift id. Shift only counts in ``shift_mode="held"``, Fn only in ``fn_mode="held"``."""
        p = self.sim.params
        mods = []
        if "Fn" in item[2] and p.fn_mode == "held":
            mods.append("Fn")
        if item[1] and p.shift_mode == "held":
            mods.append(next((m for m in item[2] if m in SHIFT_IDS), "Shift"))
        return mods

    def _fn_tap_pending(self, item) -> bool:
        return "Fn" in item[2] and self.sim.params.fn_mode == "latch" and not self.sim.keyboard.fn_latched

    def _current_chord(self):
        """The current item when it must be pressed as a two-foreleg chord, else None (single-leg paths)."""
        if self.queue is None and self.sim.params.shift_mode != "held":
            return None
        item = self._current_item()
        if item is None or not self._hold_mods(item) or self._fn_tap_pending(item):
            return None
        return item

    def _planner(self) -> Planner:
        if self._reach_planner is None:
            self._reach_planner = Planner()
        return self._reach_planner

    def _reach_table(self, leg: str) -> np.ndarray:
        """IK shortfall for a claw target ``(dx, dy)`` from the leg's press anchor (world axes, ``dx`` in [-3, 3] and
        ``dy`` in [-2, 2], the claw action's reach box), the worse of strike height and hover height. Real reach is
        smaller than the box (forward and outward corners fail), so chord planning uses this instead of the ellipse."""
        if not _REACH_TABLES:
            planner = self._planner()
            body = np.array([0.0, -2.4, STAND_HEIGHT, FACE_YAW, 0.0, 0.0])
            r = self.sim.params.tip_radius
            base = self.ik.tips(body, self.ik.rest)
            for name in TYPERS:
                i = LEGS.index(name)
                anchor = planner.body_point(body, PRESS_ANCHOR[name])
                table = np.zeros((25, 17))
                for ix, dx in enumerate(np.arange(-3.0, 3.01, 0.25)):
                    for iy, dy in enumerate(np.arange(-2.0, 2.01, 0.25)):
                        worst = 0.0
                        for z in (-0.1, HOVER):
                            target = base.copy()
                            target[i] = [anchor[0] + dx, anchor[1] + dy, z + r]
                            q = self.ik.solve(body, target, self.ik.rest.copy(), iterations=30)
                            # ``mn`` joint targets are limited to rest +- JOINT_SCALE; keep both action modes consistent
                            q[i] = np.clip(q[i], self.ik.rest[i] - JOINT_SCALE, self.ik.rest[i] + JOINT_SCALE)
                            worst = max(worst, float(np.linalg.norm(self.ik.tips(body, q)[i] - target[i])))
                        table[ix, iy] = worst
                _REACH_TABLES[name] = table
        return _REACH_TABLES[leg]

    def _leg_reachable(self, leg: str, rel: np.ndarray) -> bool:
        gx, gy = (float(rel[0]) + 3.0) / 0.25, (float(rel[1]) + 2.0) / 0.25
        if not (0.0 <= gx <= 24.0 and 0.0 <= gy <= 16.0):
            return False
        table = self._reach_table(leg)
        x0, y0 = int(np.floor(gx)), int(np.floor(gy))
        block = table[x0:min(x0 + 2, 25), y0:min(y0 + 2, 17)]
        return bool(block.max() <= CHORD_TOL)

    def _pair_ratio(self, key_id: str, mod_id: str, mod_leg: str, key_leg: str, xy) -> float:
        """Reach comfort of a two-leg pose: the worse elliptical reach ratio (1 = edge of the reach ellipse), or
        ``inf`` when either press point is outside what the leg's IK can actually reach."""
        planner, body = self._planner(), self._planner().body_at(np.asarray(xy, dtype=float))
        worst = 0.0
        for leg, target, weight in ((mod_leg, mod_id, 1.0), (key_leg, key_id, 1.5)):   # the struck key is the fussy one
            anchor = planner.body_point(body, PRESS_ANCHOR[leg])
            point = press_point(target, float(anchor[0]))
            if not self._leg_reachable(leg, point - anchor):
                return math.inf
            worst = max(worst, weight * planner.reach_ratio(body, leg, point))
        return worst

    def chord_options(self, key_id: str, mod: str) -> list:
        """Every ``(ratio, modifier id, modifier leg, key leg, thorax xy)`` from which both forelegs reach their keys
        (comfort ratio; only poses both legs can reach by IK). ``mod`` is ``"Fn"``, ``"Shift"`` (either key) or a Shift key id. Cached."""
        cache_key = (key_id, mod)
        if cache_key not in self._chord_cache:
            ids = SHIFT_IDS if mod == "Shift" else (mod,)
            (x0, x1), (y0, y1) = CHORD_BOUNDS
            options = []
            for mod_id in ids:
                for mod_leg, key_leg in (("LF", "RF"), ("RF", "LF")):
                    for x in np.arange(x0, x1 + 1e-9, CHORD_GRID):
                        for y in np.arange(y0, y1 + 1e-9, CHORD_GRID):
                            ratio = self._pair_ratio(key_id, mod_id, mod_leg, key_leg, (x, y))
                            if ratio < math.inf:
                                options.append((ratio, mod_id, mod_leg, key_leg, np.array([x, y])))
            self._chord_cache[cache_key] = options
        return self._chord_cache[cache_key]

    def chord_span(self, key_id: str, mod: str) -> float:
        """Horizontal distance (key pitches) between the modifier's and the key's nearest press points."""
        near = float(press_point(key_id, 0.0)[0])
        return min(abs(float(press_point(m, near)[0]) - float(press_point(key_id, float(press_point(m, near)[0]))[0]))
                   for m in (SHIFT_IDS if mod == "Shift" else (mod,)))

    def check_chord(self, item) -> None:
        """Raise :class:`UnreachableChord` when ``item`` cannot be realised with two held forelegs."""
        mods = self._hold_mods(item)
        if len(mods) > 1:
            raise UnreachableChord(f"{'+'.join(mods)}+{item[0]}: three keys down at once, only two forelegs press")
        if mods and not self.chord_options(item[0], mods[0]):
            raise UnreachableChord(f"{mods[0]}+{item[0]}: no thorax pose within the thorax bounds puts a foreleg on each "
                                   f"key (press points {self.chord_span(item[0], mods[0]):.1f} key pitches apart; two "
                                   f"forelegs span at most about 11)")

    def _chord_plan(self, item) -> dict:
        """Modifier leg, key leg and thorax goal for ``item``; keeps a modifier that a leg already holds if possible."""
        kb = self.sim.keyboard
        index = self.qi if self.queue is not None else len(self.buffer)
        signature = (index, tuple(sorted(kb.down.items())))
        if self._chord_plan_cache is not None and self._chord_plan_cache[0] == signature:
            return self._chord_plan_cache[1]
        key_id, mod = item[0], self._hold_mods(item)[0]
        options = self.chord_options(key_id, mod)
        if not options:
            self.check_chord(item)
        here = self.body[:2]
        best = None
        for ratio, mod_id, mod_leg, key_leg, xy in options:
            if kb.holding(mod_leg) == mod_id:      # already down: stay put if both keys are still reachable
                now = self._pair_ratio(key_id, mod_id, mod_leg, key_leg, here)
                if now < math.inf and (best is None or now < best[0]):
                    best = (now, mod_id, mod_leg, key_leg, here.copy())
        if best is None:
            best = min(options, key=lambda o: 0.15 * float(np.linalg.norm(o[4] - here)) + 3.0 * o[0])   # comfort first
        plan = {"mod": best[1], "mod_leg": best[2], "key_leg": best[3], "goal": best[4]}
        self._chord_plan_cache = (signature, plan)
        return plan

    def next_key(self) -> str | None:
        if self.queue is not None or self.sim.params.shift_mode == "held":
            item = self._current_item()
            if item is None:
                return None
            if self._fn_tap_pending(item):
                return "Fn"
            if item[1] and self.sim.params.shift_mode == "held":
                return item[0]           # held Shift: the chord's key; the Shift foreleg is the expert's business
            return physical_key_id(item[0], item[1], self.sim.keyboard.shift_latched)
        if len(self.buffer) >= len(self.target):
            return None
        return physical_key(self.target[len(self.buffer)], self.sim.keyboard.shift_latched)

    def key_point(self, leg: str = "RF") -> np.ndarray:
        key_id = self.next_key()
        return press_point(key_id, float(self.anchor_world(leg)[0])) if key_id else self.anchor_world(leg)

    def holding(self, leg: str = "RF") -> bool:
        return self.sim.keyboard.holding(leg) is not None

    def rf_holding(self) -> bool:
        return self.holding("RF")

    def _potential(self) -> float:
        leg = self.typing_leg()
        tip, point = self.tips[LEGS.index(leg)], self.key_point(leg)
        dist = float(np.linalg.norm(point - tip[:2]))
        phi = -0.2 * float(np.linalg.norm(point - self.anchor_world(leg))) - 0.3 * dist
        if self.next_key() and hit_test(*to_layout(tip[0], tip[1])) == self.next_key():
            phi += 0.5 * ((HOVER - tip[2]) if not self.holding(leg) else (tip[2] - self.z_low))
        return phi

    def _leg_obs(self, leg: str) -> list[float]:
        i = LEGS.index(leg)
        point, anchor, tip = self.key_point(leg), self.anchor_world(leg), self.tips[i]
        return [(point[0] - anchor[0]) / 8.0, (point[1] - anchor[1]) / 4.0,
                (point[0] - tip[0]) / 0.5, (point[1] - tip[1]) / 0.5,
                2 * (tip[2] - self.z_low) / (HOVER + 0.1 - self.z_low) - 1, np.clip(self.tip_vz[i] / 0.1, -1, 1)]

    def _claw_coords(self, leg: str) -> list[float]:
        """Commanded claw position in the leg's claw-action coordinates (rx, ry, rz), from forward kinematics
        of the commanded joint targets. Used in ``mn`` mode where the action is not a claw target."""
        tip = self._cmd_tips[LEGS.index(leg)]
        rel = tip[:2] - self.anchor_world(leg)
        z = tip[2] - self.sim.params.tip_radius
        return [*np.clip(rel / REACH_XY, -1, 1), float(np.clip(2 * (z - self.z_low) / (HOVER - self.z_low) - 1, -1, 1))]

    def _prev(self) -> np.ndarray:
        """The 'previous action' the observation and the expert see (claw-space 8-vector in both modes)."""
        if self.action_mode == "claw":
            return self.prev_action
        return np.array([*self.prev_action[:2], *self._claw_coords("RF"), *self._claw_coords("LF")], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        prev = self._prev()
        key_id = self.next_key()
        depression = float(self.sim.depression()[self.sim.key_ids.index(key_id)]) if key_id else 0.0
        char = self.target[len(self.buffer)] if key_id else None
        shifted_key = bool(self.queue[self.qi][1]) if self.queue is not None and key_id else \
            bool(char) and CHAR_TO_KEY[char][1]
        (x0, x1), (y0, y1) = BOUNDS
        obs = np.array([
            *self._leg_obs("RF"),
            depression / KEY_TRAVEL, float(self.holding("RF")),
            float(shifted_key), float(self.sim.keyboard.shift_latched),
            len(self.buffer) / len(self.target), 1 - self.tick / self.max_steps, self.last_correct,
            self.velocity[0] / self.max_speed, self.velocity[1] / self.max_speed,
            2 * (self.body[0] - x0) / (x1 - x0) - 1, 2 * (self.body[1] - y0) / (y1 - y0) - 1,
            *prev[:5],
            *self._leg_obs("LF"), float(self.holding("LF")), *prev[5:],
            1.0 if self.typing_leg() == "LF" else -1.0,
        ], dtype=np.float32)
        if not self.state_assist:
            obs[[self.OBS_LABELS.index(name) for name in ASSIST_LABELS]] = 0.0
        extra = []
        if self.key_obs:
            onehot = np.zeros(len(KEY_IDS), dtype=np.float32)
            if key_id:
                onehot[KEY_IDS.index(key_id)] = 1.0
            extra.append(onehot)
        if self.editor_obs:
            tail = [ord(c) / 127.0 * 2 - 1 for c in self.editor.text[-3:][::-1]]
            extra.append(np.array([self.editor.cursor / max(1, len(self.target)), *tail, *[0.0] * (3 - len(tail))],
                                  dtype=np.float32))
        if extra:
            obs = np.concatenate([obs, *extra])
        return np.clip(obs, -1.0, 1.0)

    # -- gym API --------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        keys = options.get("keys")
        target = options.get("target") if keys is None else "\0"
        if target is None:
            target = self.targets[int(self.np_random.integers(len(self.targets)))]
        if keys is not None:
            if not len(keys):
                raise ValueError("keys must be non-empty")
            self.set_keys(keys)
            target = self.target
        elif not isinstance(target, str) or not target or any(c not in CHAR_TO_KEY for c in target):
            raise ValueError("target must be a non-empty string using supported keyboard characters")
        if keys is None:
            self.queue = None
            self._wide = self.sim.params.shift_mode == "held"
        self.target, self.buffer, self.typed = target, "", ""
        self.qi = 0
        self.editor = Editor()
        self._chord_plan_cache = None
        self.tick, self.last_correct, self.tip_vz = 0, 0.0, np.zeros(len(LEGS))
        self.max_steps = int(options.get("max_steps") or self.base_steps + self.steps_per_char * len(target))
        self.planner = Planner()
        start = options.get("start")
        if start is None:
            # Anywhere the forelegs are over the keys (number row to space bar).
            start = (self.np_random.uniform(-4.0, 4.0), self.np_random.uniform(-4.4, -0.4))
        self.body = np.array([start[0], start[1], STAND_HEIGHT, FACE_YAW, 0.0, 0.0])
        self.planner.body = self.body.copy()
        feet = np.array([self.planner.nominal_world(self.body, leg) for leg in LEGS])
        for leg, i in zip(TYPERS, TYPER_INDEX):
            feet[i] = [*self.anchor_world(leg), HOVER]
        self.gait = OnlineGait(self.planner, feet)
        self.velocity = np.zeros(2)
        # Typing-leg targets relative to their anchors (body frame offsets, world axes).
        self.rel = {leg: np.array([0.0, 0.0, HOVER]) for leg in TYPERS}
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.q = self.ik.solve(self.body, self._physical(feet), self.ik.rest.copy(), iterations=20)
        self.sim.reset(self.body, self.q)
        self.tips = self.sim.tips()
        self._cmd_tips = self.ik.tips(self.body, self.q) if self.action_mode == "mn" else self.tips
        self.events, self.rows = [], []
        if self.record:
            self._record_row("stand")
        return self._observation(), {"target": target, "mode": "state_assisted_physics_body"}

    def _physical(self, feet: np.ndarray) -> np.ndarray:
        """Claw-centre targets: stance claws rest on the highest surface under them."""
        r = self.sim.params.tip_radius
        out = feet.copy()
        for i, (x, y, z) in enumerate(feet):
            if i in TYPER_INDEX:
                out[i, 2] = z + r
            elif abs(z - ground(x, y)) < 1e-9:
                out[i, 2] = support_height(x, y, r) + r
            else:
                out[i, 2] = z + r
        return out

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be a finite {self.action_dim}-vector")
        w = self.reward_weights
        phi = self._potential()
        shift_needed = bool(self.next_key()) and self.next_key() in {"ShiftLeft", "ShiftRight"}
        wanted: set[str] = set()      # modifier keys the current press wants held (chords)
        item = self._current_item()
        if item is not None:
            for spec in self._hold_mods(item):
                wanted.update(SHIFT_IDS if spec == "Shift" else (spec,))

        # Body: smoothed velocity command, clamped to the deck.
        command = action[:2] * self.max_speed
        norm = float(np.linalg.norm(command))
        if norm > self.max_speed:
            command *= self.max_speed / norm
        self.velocity += 0.5 * (command - self.velocity)
        (x0, x1), (y0, y1) = CHORD_BOUNDS if self._wide else BOUNDS
        moved = np.clip(self.body[:2] + self.velocity, (x0, y0), (x1, y1))
        self.velocity = moved - self.body[:2]    # no velocity into the deck edge
        self.body[:2] = moved
        self.planner.body = self.body.copy()
        feet = self.gait.step(self.body, self.velocity)

        feet = feet.copy()
        if self.action_mode == "claw":
            # Typing legs: absolute targets in their reach boxes, rate limited.
            # Rate limit in the body frame so each reach target travels with the thorax.
            for k, (leg, i) in enumerate(zip(TYPERS, TYPER_INDEX)):
                a = action[2 + 3 * k:5 + 3 * k]
                anchor = self.anchor_world(leg)
                goal = np.array([a[0] * REACH_XY[0], a[1] * REACH_XY[1],
                                 self.z_low + (a[2] + 1) / 2 * (HOVER - self.z_low)])
                self.rel[leg] = self.rel[leg] + np.clip(goal - self.rel[leg], -self.max_rate, self.max_rate)
                feet[i] = [anchor[0] + self.rel[leg][0], anchor[1] + self.rel[leg][1], self.rel[leg][2]]
            self.q = self.ik.solve(self.body, self._physical(feet), self.q, iterations=self.ik_iterations)
        else:
            # Four walking legs by IK from the online gait; the two typing legs take the policy's joint targets
            # directly (rest + a * JOINT_SCALE, rate limited per tick). No IK on the typers.
            walk = self.ik_walk.solve(self.body, self._physical(feet), self.q, iterations=self.ik_iterations)
            goal = self.joint_rest + action[2:].reshape(2, 7) * JOINT_SCALE
            typers = self.q[TYPER_INDEX] + np.clip(goal - self.q[TYPER_INDEX], -self.max_joint_rate, self.max_joint_rate)
            self.q = walk
            self.q[TYPER_INDEX] = typers

        before_z = self.tips[:, 2].copy()
        emitted = self.sim.step(self.q, self.body)
        self.tick += 1
        self.tips = self.sim.tips()
        self.tip_vz = self.tips[:, 2] - before_z

        reward = w["time"]
        self.last_correct = 0.0
        error = False
        for event in emitted:
            if event["type"] == "key":
                event = annotate_key_event(self.editor, dict(event))
                self.typed = self.editor.text
            self.events.append(event)
            if event["type"] == "modifier":
                reward += w["shift"] if (shift_needed or event["key_id"] in wanted) else -w["shift"]
            if event["type"] != "key":
                continue
            if self.queue is not None:
                want = self.queue[self.qi] if self.qi < len(self.queue) else None
                mods = event.get("modifiers", [])
                good = want is not None and event["key_id"] == want[0] and want[1] == any(
                    m in ("ShiftLeft", "ShiftRight") for m in mods) and ("Fn" in want[2]) == ("Fn" in mods)
                if good:
                    self.qi += 1
                    self.buffer += "\0"
            else:
                expected = self.target[len(self.buffer)] if len(self.buffer) < len(self.target) else None
                good = event["char"] == expected and event["key_id"] != "Backspace"
                if good:
                    self.buffer += event["char"]
            if good:
                reward += w["correct"]
                self.last_correct = 1.0
            else:
                reward += w["wrong"]
                error = True
        if error and not self.terminate_on_error and self.queue is None:
            self.buffer = self.typed
        exact = (self.qi >= len(self.queue) and not error) if self.queue is not None else self.typed == self.target
        terminated = exact or (error and self.terminate_on_error)
        truncated = self.tick >= self.max_steps and not terminated
        if exact:
            reward += w["complete"]
        # Zero terminal potential only on success: zeroing it after an error would
        # pay out -phi (> 0 far from the key) and make typos profitable.
        next_phi = 0.0 if exact else self._potential()
        # Undiscounted difference: with gamma < 1 and phi < 0, (gamma*phi - phi) > 0 would
        # reward standing still. The sum still telescopes to phi_end - phi_start.
        reward += w["shaping"] * (next_phi - phi)
        self.prev_action = action.astype(np.float32)
        if self.action_mode == "mn":
            self._cmd_tips = self.ik.tips(self.body, self.q)
        if self.record:
            self._record_row("press" if self.tips[TYPER_INDEX, 2].min() < HOVER * 0.5 else "walk")
        return self._observation(), float(reward), terminated, truncated, {
            "target": self.target, "text": self.typed, "events": emitted, "tick": self.tick,
            "exact": exact, "error": error,
        }

    # -- scripted expert (demonstrations and baseline) --------------------
    def expert_action(self) -> np.ndarray:
        """Scripted controller that is a function of observable state only (clonable).

        The key goes to the foreleg on its side of the body. Walk until the
        press point is inside that leg's reach box, aim its claw at it, and set
        claw height from the claw's margin inside the target keycap: hover near
        or off the edge, strike well inside, keep pushing while the key moves,
        lift while the leg holds a key. A leg that has just pressed lifts in
        place before it moves on, and the body waits for it, so no claw drags
        across keycaps. The idle foreleg hovers over its anchor.
        """
        action = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        prev = self._prev()
        key_id = self.next_key()
        if key_id is None:
            return self._expert_output(action)
        chord = self._current_chord()
        if chord is not None:
            return self._expert_output(self._chord_expert(chord))
        if self._latched_fn_item():
            return self._expert_output(self._single_expert(key_id))
        typer = self.typing_leg()
        busy = False
        for k, leg in enumerate(TYPERS):
            block = slice(2 + 3 * k, 5 + 3 * k)
            tip = self.tips[LEGS.index(leg)]
            lx, ly = to_layout(float(tip[0]), float(tip[1]))
            over = hit_test(lx, ly) == key_id and leg == typer
            if self.holding(leg) or (tip[2] < 0.5 * HOVER and not over):
                # Just pressed (holding, or low and off the target): lift in place.
                action[block] = [*prev[block][:2], 1.0]
                busy = True
                continue
            if leg != typer:
                continue
            offset = press_point(key_id, float(self.anchor_world(leg)[0])) - self.anchor_world(leg)
            if float(np.linalg.norm(offset / REACH_XY)) > 0.5:
                action[:2] = np.clip(offset / self.max_speed * 0.6, -1, 1)
            action[block][:2] = np.clip(offset / REACH_XY, -1, 1)
            key = BY_ID[key_id]
            margin = min(lx - key.x, key.x + key.width - lx, ly - key.y, key.y + key.height - ly)
            margin -= 2.0 * float(np.linalg.norm(self.velocity))
            pushing = self.sim.depression()[self.sim.key_ids.index(key_id)] > 0.005 and margin > 0.05
            # Hover off the key, strike once well inside its cap.
            action[block.start + 2] = -1.0 if pushing else 1.0 - 2.0 * float(np.clip((margin - 0.1) / 0.15, 0.0, 1.0))
        if busy:
            action[:2] = 0.0
        return self._expert_output(action)

    def _strike_z(self, key_id: str, tip: np.ndarray) -> float:
        """Claw height action for striking ``key_id``: hover near or off the edge, strike well inside, keep pushing."""
        lx, ly = to_layout(float(tip[0]), float(tip[1]))
        key = BY_ID[key_id]
        margin = min(lx - key.x, key.x + key.width - lx, ly - key.y, key.y + key.height - ly)
        margin -= 2.0 * float(np.linalg.norm(self.velocity))
        pushing = self.sim.depression()[self.sim.key_ids.index(key_id)] > 0.005 and margin > 0.05
        return -1.0 if pushing else 1.0 - 2.0 * float(np.clip((margin - 0.1) / 0.15, 0.0, 1.0))

    def _unregistered_press(self, key_id: str) -> bool:
        """The key is down past the press threshold but no contact was credited (a claw skidding off a cap edge, or a
        leg segment leaning on it): the claw must lift so the key re-arms, then strike again."""
        kb = self.sim.keyboard
        return key_id not in kb.down and not bool(kb.armed[self.sim.key_ids.index(key_id)])

    def _latched_fn_item(self) -> bool:
        item = self._current_item()
        return item is not None and "Fn" in item[2] and self.sim.params.fn_mode == "latch"

    def _single_expert(self, key_id: str) -> np.ndarray:
        """One press by one foreleg after walking to a pose that puts the key at the leg's anchor. Used for the Fn tap and
        the key that follows it in ``fn_mode="latch"``: the two keys can be a whole keyboard apart, so the claws stay
        over their anchors until the thorax has arrived instead of stretching to the far corner of the reach box."""
        (x0, x1), (y0, y1) = BOUNDS
        planner = self._planner()
        plans = []
        for leg in TYPERS:
            goal = planner.anchor_goal(BY_ID[key_id], leg)
            goal = np.array([min(max(goal[0], x0), x1), min(max(goal[1], y0), y1)])
            ratio = self._pair_ratio(key_id, key_id, leg, leg, goal)
            plans.append((ratio == math.inf, float(np.linalg.norm(goal - self.body[:2])), leg, goal))
        _, _, typer, goal = min(plans, key=lambda p: p[:2])
        offset_body = goal - self.body[:2]
        settled = float(np.linalg.norm(offset_body)) < 0.12
        kb = self.sim.keyboard
        action = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        prev = self._prev()
        busy = False
        for k, leg in enumerate(TYPERS):
            block = slice(2 + 3 * k, 5 + 3 * k)
            tip = self.tips[LEGS.index(leg)]
            over = hit_test(*to_layout(float(tip[0]), float(tip[1]))) == key_id and leg == typer
            if kb.holding(leg) is not None or (tip[2] < 0.5 * HOVER and not over):
                action[block] = [*prev[block][:2], 1.0]
                busy = True
            elif leg == typer and settled:
                offset = press_point(key_id, float(self.anchor_world(leg)[0])) - self.anchor_world(leg)
                action[block][:2] = np.clip(offset / REACH_XY, -1, 1)
                action[block.start + 2] = self._strike_z(key_id, tip)
        if busy:
            action[:2] = 0.0
        elif not settled:
            action[:2] = np.clip(offset_body / self.max_speed * 0.8, -1, 1)
        return action

    def _chord_expert(self, item) -> np.ndarray:
        """Two-foreleg chord: walk to a pose where both forelegs reach, press and hold the modifier with one, strike
        the key with the other, then lift both (the modifier stays down when the next press shares it and is still
        reachable, which is how a Shift selection keeps Shift held across several arrows)."""
        plan = self._chord_plan(item)
        key_id, mod_id, mod_leg, key_leg = item[0], plan["mod"], plan["mod_leg"], plan["key_leg"]
        kb = self.sim.keyboard
        goal_offset = plan["goal"] - self.body[:2]
        # A modifier held from the previous press stays down only if the thorax need not move; otherwise it is lifted
        # (below, as a leg "holding" a key) and pressed again from the new pose.
        holding_mod = kb.holding(mod_leg) == mod_id and float(np.linalg.norm(goal_offset)) < 0.15
        settled = holding_mod or float(np.linalg.norm(goal_offset)) < 0.12
        action = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        prev = self._prev()
        busy = False
        for k, leg in enumerate(TYPERS):
            block = slice(2 + 3 * k, 5 + 3 * k)
            tip = self.tips[LEGS.index(leg)]
            target = mod_id if leg == mod_leg else key_id
            over = hit_test(*to_layout(float(tip[0]), float(tip[1]))) == target
            stuck = leg == key_leg and holding_mod and self._unregistered_press(key_id)
            if not (leg == mod_leg and holding_mod) and (kb.holding(leg) is not None or stuck
                                                         or (tip[2] < 0.5 * HOVER and not over)):
                action[block] = [*prev[block][:2], 1.0]      # just pressed (or a stray touch): lift in place
                busy = True
                continue
            if not settled:
                continue      # claws wait over their anchors while the thorax walks (a far reach dips into keys)
            offset = press_point(target, float(self.anchor_world(leg)[0])) - self.anchor_world(leg)
            action[block][:2] = np.clip(offset / REACH_XY, -1, 1)
            if leg == mod_leg:
                action[block.start + 2] = -1.0 if holding_mod else (self._strike_z(target, tip) if settled else 1.0)
            else:
                action[block.start + 2] = self._strike_z(target, tip) if (holding_mod and settled) else 1.0
        if busy or holding_mod:
            action[:2] = 0.0
        elif float(np.linalg.norm(goal_offset)) > 0.08:
            action[:2] = np.clip(goal_offset / self.max_speed * 0.8, -1, 1)
        return action

    def _expert_output(self, claw: np.ndarray) -> np.ndarray:
        """Expert action in the env's action space: the claw action itself (``claw`` mode), or in ``mn`` mode the
        thorax velocity plus the 14 joint angles that ``LegIK`` gives for the rate-limited claw target."""
        if self.action_mode == "claw":
            return claw.astype(np.float32)
        feet = np.array([self.gait.feet[i] for i in range(len(LEGS))])
        for k, (leg, i) in enumerate(zip(TYPERS, TYPER_INDEX)):
            a = claw[2 + 3 * k:5 + 3 * k]
            anchor = self.anchor_world(leg)
            goal = np.array([a[0] * REACH_XY[0], a[1] * REACH_XY[1], self.z_low + (a[2] + 1) / 2 * (HOVER - self.z_low)])
            cur = np.array([*(self._cmd_tips[i][:2] - anchor), self._cmd_tips[i][2] - self.sim.params.tip_radius])
            rel = cur + np.clip(goal - cur, -self.max_rate, self.max_rate)
            feet[i] = [anchor[0] + rel[0], anchor[1] + rel[1], rel[2]]
        q = self.ik_typers.solve(self.body, self._physical(feet), self.q, iterations=self.ik_iterations)
        joints = np.clip((q[TYPER_INDEX] - self.joint_rest) / JOINT_SCALE, -1.0, 1.0)
        return np.concatenate([claw[:2], joints.ravel()]).astype(np.float32)

    # -- replay -----------------------------------------------------------
    def _record_row(self, phase: str) -> None:
        activity = synthetic_activity(self.tips, phase, float(np.linalg.norm(self.velocity)), 0.0)
        self.rows.append([self.tick, *np.round(self.body, 4).tolist(),
                          *np.round(self.sim.joint_q().ravel(), 3).tolist(),
                          *np.round(self.tips.ravel(), 3).tolist(), *activity])

    def replay(self, target_id: str, mode: str) -> dict:
        fields = ["tick", "x", "y", "z", "yaw", "pitch", "roll"]
        fields += [f"{leg}.{dof}" for leg in LEGS for dof in self.sim.fly.leg_dofs]
        fields += [f"{leg}.tip.{axis}" for leg in LEGS for axis in "xyz"]
        fields += [f"activity.{i}" for i in range(16)]
        p = self.sim.params
        from .keyboard import KEY_PITCH_MM
        return {
            "metadata": {"target_id": target_id, "mode": mode, "observation_condition": "state_assisted",
                         "tick_hz": p.tick_hz, "fly_scale": FLY_SCALE, "key_pitch_mm": KEY_PITCH_MM,
                         "body_model": "NeuroMechFly v2 tree in MuJoCo: 42 position-actuated leg joints, "
                                       "claw-sphere contacts; thorax carried by mocap, gravity off",
                         "frame": "X right, Y toward screen, Z up; key pitch units; origin keyboard centre",
                         "contact_threshold_z": -p.press_depth, "max_ik_error": 0.0,
                         "unattributed_presses": self.sim.keyboard.unattributed},
            "frames": {"fields": fields, "rows": self.rows},
            "events": self.events, "target": self.target, "final_text": self.typed,
            "final_text_hash": text_hash(self.typed),
        }
