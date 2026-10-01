"""Scripted, kinematic NeuroMechFly typing demonstration.

The laptop is fly sized (``keyboard.KEY_PITCH_MM``), so the fly stands over
several keys. It types with both forelegs: each key goes to the foreleg on its
side when that leg can reach it. Only when neither can does an open-loop
tripod gait walk the thorax (stance feet stay fixed on the keyboard, swing
feet arc to new footholds), turning toward long walks. All six legs
are posed by inverse kinematics on the NeuroMechFly v2 kinematic tree
(``backend.flybody``). Only the contact detector, fed with the *posed* claw
tips of all six legs, can emit a key; target text never writes to the editor.
No physics, joint torques, or learned policy is claimed by this baseline.

World frame (key pitches, "u"): X to the right, Y away from the user toward
the screen, Z up; the origin is the keyboard centre and Z=0 is the keycap
top surface. Fly model millimetres are scaled by ``FLY_SCALE``.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .flybody import default_body, euler_zyx
from .editor import Editor
from .keyboard import (BY_ID, CHAR_TO_KEY, KEY_PITCH_MM, KEY_TRAVEL, LAYOUT_HEIGHT, LAYOUT_WIDTH,
                       ContactKeyboard, key_char, surface_z)

TICK_HZ = 30
FLY_SCALE = 1.0                      # life-size fly; the laptop is scaled down instead
MM = FLY_SCALE / KEY_PITCH_MM        # one NeuroMechFly millimetre in key pitches
STAND_HEIGHT = 1.0 * MM              # thorax origin above the surface
FACE_YAW = math.pi / 2               # facing +Y, toward the screen
GAIT_HALF = 5                        # ticks per tripod swing (3 Hz stepping at 30 Hz)
SWING_HEIGHT = 0.3 * MM
HOVER = 0.32                         # foreleg hover above the key before a press
MAX_SPEED = 4.0 * MM / TICK_HZ       # u per tick (4 mm/s)
RAMP_TICKS = 6
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
TRIPODS = ({"LF", "RM", "LH"}, {"RF", "LM", "RH"})
PRESS_ANCHOR = {"LF": np.array([0.85, 0.75]), "RF": np.array([0.85, -0.75])}   # fly mm, thorax frame
REACH = np.array([0.6, 0.9])         # fly mm, forward/lateral half-axes of the reach ellipse around an anchor
LEAN = 0.3 * MM                      # thorax shift over planted feet before the fly has to step
LOOKAHEAD = 10                       # upcoming keys a new standing spot should cover
BOUNDS = ((-3.2, 3.2), (-4.5, 0.5))  # thorax XY limits that keep all six feet on the laptop
START_XY = (0.0, -2.4)               # forelegs over the home row


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def apply_key(buffer: str, key_id: str, char: str) -> str:
    """Legacy string interface: the key applied to ``buffer`` with the cursor at its end."""
    return Editor(buffer).apply(key_id, (), char=char).text


def annotate_key_event(editor: Editor, event: dict) -> dict:
    """Apply a ``key`` event to ``editor`` and add the C4 editor fields to it (in place)."""
    mods = event.get("modifiers", [])
    result = editor.apply(event["key_id"], mods, char=event["char"]).as_dict()
    result.pop("char")
    result.pop("key")
    event.update(result)
    event["key"] = event["key_id"]
    event["modifiers"] = list(mods)
    return event


def to_layout(x: float, y: float) -> tuple[float, float]:
    return x + LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2 - y


def to_world(x: float, y: float) -> tuple[float, float]:
    return x - LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2 - y


def ground(x: float, y: float) -> float:
    return surface_z(*to_layout(x, y))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3 - 2 * s)


class Planner:
    """Produces per-tick thorax poses and foot targets; knows nothing about text."""

    def __init__(self) -> None:
        fly = default_body()
        self.nominal = {leg: fly.tip(leg, fly.rest_q[leg])[:2] for leg in LEGS}
        self.body = np.array([*START_XY, STAND_HEIGHT, FACE_YAW, 0.0, 0.0])
        self.feet = np.array([self.nominal_world(self.body, leg) for leg in LEGS])
        self.hover_leg: str | None = None
        self.stance_xy = self.body[:2].copy()   # thorax position the planted feet were placed for
        self.bodies: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.phase: list[str] = []

    # -- geometry -----------------------------------------------------------
    def body_point(self, body: np.ndarray, xy_mm: np.ndarray) -> np.ndarray:
        """World XY of a thorax-frame point (fly mm), ignoring pitch/roll."""
        c, s = math.cos(body[3]), math.sin(body[3])
        x, y = xy_mm * MM
        return body[:2] + np.array([c * x - s * y, s * x + c * y])

    def nominal_world(self, body: np.ndarray, leg: str) -> np.ndarray:
        x, y = self.body_point(body, self.nominal[leg])
        return np.array([x, y, ground(x, y)])

    def record(self, phase: str) -> None:
        self.bodies.append(self.body.copy())
        self.targets.append(self.feet.copy())
        self.phase.append(phase)

    # -- behaviours ---------------------------------------------------------
    def idle(self, ticks: int) -> None:
        for _ in range(ticks):
            self.record("stand")

    def walk_to(self, goal_xy: np.ndarray) -> None:
        goal = np.array([min(max(goal_xy[0], BOUNDS[0][0]), BOUNDS[0][1]),
                         min(max(goal_xy[1], BOUNDS[1][0]), BOUNDS[1][1])])
        start = self.body.copy()
        delta = goal - start[:2]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-3 and abs(wrap(start[3] - FACE_YAW)) < 1e-3:
            return
        # Smooth trapezoidal speed profile.
        ticks = max(GAIT_HALF * 2, math.ceil(distance / MAX_SPEED) + RAMP_TICKS)
        speed = [smoothstep((k + 0.5) / RAMP_TICKS) * smoothstep((ticks - k - 0.5) / RAMP_TICKS) for k in range(ticks)]
        cumulative = np.cumsum(speed) / sum(speed)
        # Long walks turn toward the direction of travel, then face the screen again.
        turn = 0.0
        if distance > 2.3 * MM:
            heading = wrap(math.atan2(delta[1], delta[0]) - FACE_YAW)
            turn = max(-1.75, min(1.75, heading)) * min(1.0, (distance - 2.3 * MM) / (2.9 * MM))
        start_yaw_error = wrap(start[3] - FACE_YAW)
        path = []
        for k, s in enumerate(cumulative):
            u = (k + 1) / ticks
            window = smoothstep(u / 0.3) * smoothstep((1 - u) / 0.3)
            yaw = FACE_YAW + start_yaw_error * (1 - smoothstep(u)) + turn * window
            path.append(np.array([*(start[:2] + delta * s), STAND_HEIGHT, yaw, 0.0, 0.0]))

        swings: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        # The group holding a raised foreleg steps first so it lands promptly.
        group = 1 if self.hover_leg in TRIPODS[1] else 0
        quiet_halves, k = 0, 0
        while True:
            if k % GAIT_HALF == 0:
                if k >= ticks and quiet_halves >= 2:
                    break
                moved = False
                lookahead = path[min(len(path) - 1, k + GAIT_HALF + GAIT_HALF // 2)]
                for i, leg in enumerate(LEGS):
                    if leg not in TRIPODS[group]:
                        continue
                    target = self.nominal_world(lookahead, leg)
                    if leg == self.hover_leg or np.linalg.norm(target[:2] - self.feet[i, :2]) > 0.04:
                        swings[i] = (self.feet[i].copy(), target, k)
                        moved = True
                quiet_halves = 0 if moved or k < ticks else quiet_halves + 1
                if self.hover_leg in TRIPODS[group]:
                    self.hover_leg = None
                group = 1 - group
                if k > ticks + 8 * GAIT_HALF:
                    break
            self.body = path[min(k, len(path) - 1)].copy()
            for i, (lift, land, t0) in list(swings.items()):
                s = (k - t0 + 1) / GAIT_HALF
                xy = lift[:2] + (land[:2] - lift[:2]) * smoothstep(s)
                # sin^2 arc: zero vertical speed at lift-off and touchdown (soft landings).
                z = lift[2] + (land[2] - lift[2]) * smoothstep(s) + SWING_HEIGHT * math.sin(math.pi * min(1.0, s)) ** 2
                self.feet[i] = [xy[0], xy[1], z]
                if s >= 1:
                    self.feet[i] = land
                    del swings[i]
            self.record("walk")
            k += 1
        self.stance_xy = self.body[:2].copy()

    def lean_to(self, goal_xy: np.ndarray) -> None:
        """Shift the thorax over the planted feet (no steps); a raised foreleg moves with it."""
        start = self.body.copy()
        raised = LEGS.index(self.hover_leg) if self.hover_leg else None
        foot = self.feet[raised].copy() if raised is not None else None
        ticks = max(3, math.ceil(float(np.linalg.norm(goal_xy - start[:2])) / 0.12))
        for k in range(1, ticks + 1):
            self.body = start.copy()
            self.body[:2] = start[:2] + (goal_xy - start[:2]) * smoothstep(k / ticks)
            if raised is not None:
                self.feet[raised, :2] = foot[:2] + self.body[:2] - start[:2]
            self.record("lean")

    def lower_leg(self, leg: str) -> None:
        """Return a raised foreleg to its stance foothold."""
        i = LEGS.index(leg)
        lift, land = self.feet[i].copy(), self.nominal_world(self.body, leg)
        for k in range(1, 5):
            s = k / 4
            xy = lift[:2] + (land[:2] - lift[:2]) * smoothstep(s)
            self.feet[i] = [xy[0], xy[1], lift[2] + (land[2] - lift[2]) * smoothstep(s)]
            self.record("lower")
        self.feet[i] = land
        self.hover_leg = None

    def press(self, leg: str, point: np.ndarray) -> None:
        i = LEGS.index(leg)
        start = self.feet[i].copy()
        above = np.array([point[0], point[1], HOVER])
        base = self.body.copy()
        # Long reaches across the keys take longer (about 0.3 u per tick), arc higher,
        # and settle over the key, so a lagging physical claw clears its neighbours.
        distance = float(np.linalg.norm(above[:2] - start[:2]))
        reach = max(4 if self.hover_leg == leg else 6, math.ceil(distance / 0.3))
        arc = 0.06 + 0.1 * distance
        # Lift (or glide while raised) to hover above the key.
        for k in range(1, reach + 1):
            s = smoothstep(k / reach)
            xy = start[:2] + (above[:2] - start[:2]) * s
            z = start[2] + (HOVER - start[2]) * s + arc * math.sin(math.pi * k / reach)
            self.feet[i] = [xy[0], xy[1], z]
            self.record("reach")
        for _ in range(2 if distance > 1.0 else 0):
            self.record("reach")
        # Strike: touch the cap, push it to the bottom of its travel, hold, release.
        strike = [0.12, 0.01, -KEY_TRAVEL * 0.6, -KEY_TRAVEL, -KEY_TRAVEL, -KEY_TRAVEL * 0.5, 0.12, HOVER * 0.8, HOVER]
        lean = [0.3, 0.7, 1.0, 1.0, 1.0, 0.8, 0.4, 0.15, 0.0]
        for z, w in zip(strike, lean):
            self.feet[i] = [point[0], point[1], z]
            # Dip toward the pressing leg; tilt is sized so the hind feet move a similar
            # distance at any fly-to-keyboard scale.
            self.body = base + np.array([0, 0, -0.05 * w, 0, 0.05 / MM * w, (0.02 if leg[0] == "R" else -0.02) / MM * w])
            self.record("press")
        self.body = base
        self.hover_leg = leg

    def reach_ratio(self, body: np.ndarray, leg: str, point: np.ndarray) -> float:
        """<= 1 when ``point`` is inside ``leg``'s reach ellipse for this thorax pose."""
        return float(np.linalg.norm((self.to_body(body, point) - PRESS_ANCHOR[leg]) / REACH))

    def body_at(self, xy: np.ndarray) -> np.ndarray:
        body = self.body.copy()
        body[:2] = xy
        return body

    def anchor_goal(self, key, leg: str) -> np.ndarray:
        """Thorax XY (facing the screen) that puts ``key`` at ``leg``'s press anchor."""
        point = self.press_point(key, self.body_point(self.body, PRESS_ANCHOR[leg]))
        c, s = math.cos(FACE_YAW), math.sin(FACE_YAW)
        x, y = PRESS_ANCHOR[leg] * MM
        return point - np.array([c * x - s * y, s * x + c * y])

    def covers(self, xy: np.ndarray, key_id: str, limit: float = 1.0) -> bool:
        body, key = self.body_at(xy), BY_ID[key_id]
        return any(self.reach_ratio(body, leg, self.press_point(key, self.body_point(body, PRESS_ANCHOR[leg]))) <= limit
                   for leg in PRESS_ANCHOR)

    def standing_spot(self, key, leg: str, upcoming: tuple[str, ...]) -> np.ndarray:
        """Where to walk: ``leg`` reaches ``key`` and the most upcoming keys are in reach."""
        goal = self.anchor_goal(key, leg)
        (x0, x1), (y0, y1) = BOUNDS
        best, best_score = goal, -math.inf
        for dx in np.arange(-2.0, 2.01, 0.5):
            for dy in np.arange(-1.0, 1.01, 0.5):
                xy = np.array([min(max(goal[0] + dx, x0), x1), min(max(goal[1] + dy, y0), y1)])
                body = self.body_at(xy)
                point = self.press_point(key, self.body_point(body, PRESS_ANCHOR[leg]))
                if self.reach_ratio(body, leg, point) > 0.85:
                    continue
                score = sum(0.9 ** i for i, k in enumerate(upcoming) if self.covers(xy, k, 0.9))
                score -= 0.05 * float(np.linalg.norm(xy - self.body[:2]))
                if score > best_score:
                    best, best_score = xy, score
        return best

    def type_key(self, key_id: str, upcoming: tuple[str, ...] = ()) -> None:
        key = BY_ID[key_id]
        # Each key belongs to the foreleg on its side of the body; the other
        # foreleg covers it only when the own-side leg cannot reach.
        centre = self.press_point(key, self.body[:2])
        side = "LF" if self.to_body(self.body, centre)[1] > 0 else "RF"
        order = (side, "RF" if side == "LF" else "LF")
        choices = []
        for leg in PRESS_ANCHOR:
            point = self.press_point(key, self.body_point(self.body, PRESS_ANCHOR[leg]))
            choices.append((leg != side, self.reach_ratio(self.body, leg, point), leg))
        reachable = sorted(c for c in choices if c[1] <= 1.0)
        if reachable:
            leg = reachable[0][2]
        else:
            # Lean toward the key over the planted feet if that is enough; otherwise
            # walk to a spot that also covers the next few keys.
            leg, lean = None, None
            for candidate in order:
                goal = self.anchor_goal(key, candidate)
                for t in np.linspace(0.0, 1.0, 21)[1:]:
                    xy = self.body[:2] + (goal - self.body[:2]) * t
                    body = self.body_at(xy)
                    point = self.press_point(key, self.body_point(body, PRESS_ANCHOR[candidate]))
                    if self.reach_ratio(body, candidate, point) <= 0.9:
                        break
                if np.linalg.norm(xy - self.stance_xy) <= LEAN:
                    leg, lean = candidate, xy
                    break
            if leg is not None:
                self.lean_to(lean)
            else:
                cost = {c: float(np.linalg.norm(self.anchor_goal(key, c) - self.body[:2])) - (0.3 if c == side else 0.0)
                        for c in order}
                leg = min(order, key=cost.get)
                self.walk_to(self.standing_spot(key, leg, upcoming))
        point = self.press_point(key, self.body_point(self.body, PRESS_ANCHOR[leg]))
        if self.hover_leg is not None and self.hover_leg != leg:
            self.lower_leg(self.hover_leg)
        self.press(leg, point)

    def type_keys(self, sequence: list[str]) -> None:
        for i, key_id in enumerate(sequence):
            self.type_key(key_id, tuple(sequence[i + 1:i + 1 + LOOKAHEAD]))

    def to_body(self, body: np.ndarray, point: np.ndarray) -> np.ndarray:
        c, s = math.cos(body[3]), math.sin(body[3])
        d = (point - body[:2]) / MM
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    @staticmethod
    def press_point(key, near_xy: np.ndarray) -> np.ndarray:
        """Key centre, or for long keys the point nearest the foot, inset from the edges."""
        inset = min(0.55, key.width / 2 - 0.05)
        x0, x1 = to_world(key.x + inset, 0)[0], to_world(key.x + key.width - inset, 0)[0]
        cx = min(max(near_xy[0], x0), x1)
        cy = to_world(0, key.y + key.height / 2)[1]
        return np.array([cx, cy])


def key_sequence(target: str) -> list[str]:
    sequence = []
    for character in target:
        if character not in CHAR_TO_KEY:
            raise ValueError(f"Unsupported target character: {character!r}")
        key_id, shifted = CHAR_TO_KEY[character]
        if shifted:
            # One-shot Shift is a separate physical press; use the Shift nearer the key.
            key = BY_ID[key_id]
            left, right = BY_ID["ShiftLeft"], BY_ID["ShiftRight"]
            sequence.append("ShiftLeft" if abs(key.x - (left.x + left.width)) <= abs(right.x - key.x) else "ShiftRight")
        sequence.append(key_id)
    return sequence


def pose_legs(bodies: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Batched IK: foot targets (T, 6, 3) world -> joint angles (T, 6, 7) and posed tips (T, 6, 3)."""
    fly = default_body()
    t = len(bodies)
    rot = np.array([euler_zyx(b[3], b[4], b[5]) for b in bodies])
    local = np.einsum("tji,tlj->tli", rot, targets - bodies[:, None, :3]) / MM
    # Standing and pressing ticks repeat leg targets; solve each distinct one once.
    keys = np.concatenate([np.tile(np.arange(6), t)[:, None], np.round(local.reshape(-1, 3), 6)], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    q_unique, error = fly.solve_batch(unique[:, 0].astype(int), unique[:, 1:])
    q = q_unique[inverse.ravel()]
    tips_local, _ = fly.batch_tip_and_jacobian(np.tile(np.arange(6), t), q)
    tips = np.einsum("tij,tlj->tli", rot, tips_local.reshape(t, 6, 3) * MM) + bodies[:, None, :3]
    return q.reshape(t, 6, -1), tips, float(error.max()) * MM


def synthetic_activity(tips: np.ndarray, phase: str, speed: float, turn: float) -> list[float]:
    """16 illustrative channels driven by leg state; not measured or connectome derived."""
    heights = [min(1.0, max(0.0, z / HOVER)) for z in tips[:, 2]]
    press = min(1.0, max(0.0, -tips[:, 2].min() / KEY_TRAVEL))
    stance = [1.0 - h for h in heights]
    return [round(v, 2) for v in (*heights, *stance, min(1.0, speed / MAX_SPEED), min(1.0, abs(turn) * 10),
                                  press, 1.0 if phase == "press" else 0.2)]


def scripted_episode(target_id: str, target: str) -> dict:
    planner = Planner()
    planner.idle(4)
    planner.type_keys(key_sequence(target))
    if planner.hover_leg:
        planner.lower_leg(planner.hover_leg)
    planner.idle(6)

    bodies = np.array(planner.bodies)
    q, tips, ik_error = pose_legs(bodies, np.array(planner.targets))

    keyboard = ContactKeyboard(shift_mode="latch")   # the planner presses Shift as a separate one-shot press
    events: list[dict] = []
    rows: list[list[float]] = []
    editor = Editor()
    for tick, (body, joints, feet, phase) in enumerate(zip(bodies, q, tips, planner.phase)):
        previous = bodies[max(0, tick - 1)]
        speed = float(np.linalg.norm(body[:2] - previous[:2]))
        activity = synthetic_activity(feet, phase, speed, wrap(body[3] - previous[3]))
        rows.append([tick, *np.round(body, 4).tolist(), *np.round(joints.ravel(), 3).tolist(),
                     *np.round(feet.ravel(), 3).tolist(), *activity])
        body_pose = {"x": round(float(body[0]), 3), "y": round(float(body[1]), 3), "z": round(float(body[2]), 3),
                     "yaw": round(float(body[3]), 3)}
        for leg, (x, y, z) in zip(LEGS, feet):
            lx, ly = to_layout(float(x), float(y))
            for event in keyboard.update(tick, lx, ly, float(z), foot=leg):
                event["body"] = body_pose
                event["foot_pose"] = {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 4)}
                if event["type"] == "key":
                    annotate_key_event(editor, event)
                events.append(event)

    fields = ["tick", "x", "y", "z", "yaw", "pitch", "roll"]
    fields += [f"{leg}.{dof}" for leg in LEGS for dof in default_body().leg_dofs]
    fields += [f"{leg}.tip.{axis}" for leg in LEGS for axis in "xyz"]
    fields += [f"activity.{i}" for i in range(16)]
    return {
        "metadata": {
            "target_id": target_id,
            "mode": "scripted_kinematic_neuromechfly",
            "observation_condition": "state_assisted",
            "seed": 0,
            "tick_hz": TICK_HZ,
            "body_model": "NeuroMechFly v2 kinematic tree (FlyGym), six IK-posed legs, no physics",
            "fly_scale": FLY_SCALE,
            "key_pitch_mm": KEY_PITCH_MM,
            "frame": "X right, Y toward screen, Z up; key pitch units; origin keyboard centre",
            "contact_threshold_z": -keyboard.press_depth,
            "max_ik_error": round(ik_error, 5),
        },
        "frames": {"fields": fields, "rows": rows},
        "events": events,
        "target": target,
        "final_text": editor.text,
        "final_text_hash": text_hash(editor.text),
    }


def replay_text(events: list[dict]) -> str:
    """Reconstruct and verify text without running the controller.

    Every ``key`` event needs an unreleased contact on its foot with the same key (and the
    same ``contact_id`` when the events carry ids). C4 editor fields, when present, are
    re-derived by an :class:`Editor` and must match. Returns the final buffer.
    """
    editor = Editor()
    down: dict[str, str] = {}
    ids: dict[str, int] = {}
    emitted: set[str] = set()
    last_tick = -1
    for event in events:
        tick = event["tick"]
        if tick < last_tick:
            raise ValueError("Replay ticks must be monotonic")
        last_tick = tick
        foot = event.get("foot", "right_foreleg")
        if event["type"] == "contact_onset":
            if foot in down:
                raise ValueError("Overlapping contacts on one foot")
            if event["key_id"] in down.values():
                raise ValueError("Key already held by another foot")
            down[foot] = event["key_id"]
            ids.pop(foot, None)
            if "contact_id" in event:
                ids[foot] = event["contact_id"]
            emitted.discard(foot)
        elif event["type"] == "contact_offset":
            if down.get(foot) != event["key_id"] or ids.get(foot) != event.get("contact_id", ids.get(foot)):
                raise ValueError("Contact offset mismatch")
            del down[foot]
            ids.pop(foot, None)
        elif event["type"] == "key":
            if down.get(foot) != event["key_id"]:
                raise ValueError("Character without active key contact")
            if "contact_id" in event and ids.get(foot) != event["contact_id"]:
                raise ValueError("Key event contact_id does not match its contact")
            if foot in emitted:
                raise ValueError("Multiple characters from one contact")
            emitted.add(foot)
            mods = event.get("modifiers", [])
            if "op" in event and not (event.get("shift_latched") or event.get("fn_latched")):
                held = set(down.values()) - {event["key_id"]}
                if any(m not in held for m in mods):
                    raise ValueError("Modifier not held at contact onset")
                if event["char"] != key_char(event["key_id"], mods):
                    raise ValueError("Character does not match key and modifiers")
            result = editor.apply(event["key_id"], mods, char=event["char"])
            if event.get("text") != editor.text or event.get("text_hash") != text_hash(editor.text):
                raise ValueError("Text or hash mismatch")
            if "op" in event:
                expected = {"op": result.op, "cursor_before": result.cursor_before, "cursor_after": result.cursor_after,
                            "selection_before": result.selection_before, "selection_after": result.selection_after,
                            "buffer_hash": result.buffer_hash}
                if any(event.get(k, v) != v for k, v in expected.items()):
                    raise ValueError("Editor fields mismatch")
    return editor.text
