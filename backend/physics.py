"""MuJoCo keyboard with a physically actuated NeuroMechFly leg system.

The scene is built from the vendored NeuroMechFly v2 kinematic tree
(``data/neuromechfly/model.json``), scaled to key pitches like
``backend.embodiment``. Each of the 42 leg degrees of freedom is a hinge
joint driven by a position actuator. Every key is a spring-loaded slider.
Claw-tip spheres collide with key caps and with the keyboard deck.

A key types only when (1) it is pushed down past ``press_depth`` and (2) a
claw touched that key cap within ``touch_window`` (a tap can outrun the claw). The key re-arms after it rises
above ``release_depth``. So a character needs a physical push by a leg: the
controller cannot emit keys, and hovering or visual-only motion cannot either.

Limitations, reported with every episode: the thorax is carried kinematically
by a mocap body (the legs do not support body weight), gravity is off, and
key stiffness is a game-scale parameter chosen so a fly-scale leg can press a
key. No biological force claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .flybody import SEGMENTS, axis_angle, default_body, euler_zyx
from .keyboard import (BY_ID, DECK_DEPTH, KEY_TRAVEL, LAYOUT, LAYOUT_HEIGHT, LAYOUT_WIDTH, MODIFIER_KEYS, SHIFT_KEYS,
                       emits_key, key_char, modifier_state)

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
# Leg gains were tuned with the fly at this size (key pitches per fly mm).
TUNED_SCALE = 10.0 / 19.05


def joint_gain(scale: float) -> float:
    """Leg segment inertia grows as scale**3 (capsule mass ~ length, inertia ~ m L^2).
    Scaling kp, kv, armature, and damping by the same factor keeps the tuned leg
    dynamics at any fly-to-keyboard size."""
    return (scale / TUNED_SCALE) ** 3


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).ravel())
    return quat


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


@dataclass(frozen=True)
class PhysicsParams:
    """Frozen physics constants (game-scale; key pitch length unit, seconds)."""

    timestep: float = 0.001
    tick_hz: int = 30
    kp: float = 800.0                # leg joint position gain
    kv: float = 16.0                 # leg joint velocity gain
    armature: float = 0.01
    joint_damping: float = 0.05
    tip_radius: float = 0.015
    key_stiffness: float = 300.0     # spring force per unit key travel
    key_damping: float = 30.0
    push_through: float = 0.2         # strike target depth below the bottom of key travel
    key_preload: float = 0.1          # springref above the rest stop, so force k*preload holds keys up
    key_mass: float = 4.0            # same for every key (long keys have stabilizers)
    pad_timeconst: float = 0.004     # claw pad contact compliance (s)
    touch_window: float = 0.01      # s; a press counts if a claw touched the cap this recently
    press_depth: float = 0.05        # key travel that registers a press
    release_depth: float = 0.02      # key travel below which the key re-arms
    debounce_ticks: int = 2
    shift_mode: str = "latch"        # "latch": one-shot Shift (legacy); "held": Shift is down while its claw contact persists
    fn_mode: str = "held"            # "held": Fn is down while its contact persists (C2); "latch": an Fn tap arms the next press

    @property
    def substeps(self) -> int:
        return max(1, round(1.0 / (self.tick_hz * self.timestep)))


def key_world_center(key) -> tuple[float, float]:
    return key.x + key.width / 2 - LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2 - (key.y + key.height / 2)


def build_mjcf(params: PhysicsParams, scale: float) -> str:
    """MJCF text for the keyboard, deck, and carried fly with six actuated legs."""
    fly = default_body()
    import json
    model = json.loads(fly_model_text())
    bodies = {body["name"]: body for body in model["bodies"]}
    dof_names = {f"joint_{leg}{dof}" for leg in LEGS for dof in fly.leg_dofs}
    gain = joint_gain(scale)

    def leg_xml(leg: str) -> str:
        parts, closes = [], []
        for index, segment in enumerate(SEGMENTS):
            body = bodies[f"{leg}{segment}"]
            local = np.asarray(body["quat"], dtype=float)
            joints = body["joints"]
            fixed = [j for j in joints if j["name"] not in dof_names]
            moving = [j for j in joints if j["name"] in dof_names]
            if fixed and moving:
                raise ValueError(f"mixed fixed/moving joints in {body['name']}")
            rot = _quat_mat(local)
            for joint in fixed:   # bake fixed tarsal joints into the body frame
                rot = rot @ axis_angle(np.asarray(joint["axis"], float), joint["rest"])
            pos = np.asarray(body["pos"], float) * scale
            parts.append(f'<body name="{leg}{segment}" pos="{_fmt(pos)}" quat="{_fmt(_mat_to_quat(rot))}">')
            for joint in moving:
                parts.append(f'<joint name="{joint["name"]}" type="hinge" axis="{_fmt(joint["axis"])}" '
                             f'armature="{params.armature * gain:.6g}" damping="{params.joint_damping * gain:.6g}"/>')
            # Visual/inertial capsule toward the next segment (or the claw).
            if index + 1 < len(SEGMENTS):
                end = np.asarray(bodies[f"{leg}{SEGMENTS[index + 1]}"]["pos"], float) * scale
            else:
                end = fly.tip_offset[leg] * scale
            radius = 0.02 if index < 3 else 0.012
            parts.append(f'<geom type="capsule" fromto="0 0 0 {_fmt(end)}" size="{radius}" '
                         f'contype="0" conaffinity="0" density="300" rgba="0.55 0.35 0.2 1"/>')
            if index + 1 == len(SEGMENTS):
                parts.append(f'<geom name="{leg}_tip" type="sphere" pos="{_fmt(end)}" size="{params.tip_radius}" '
                             f'contype="1" conaffinity="2" condim="3" friction="0.8" density="300" solref="{params.pad_timeconst} 1" solmix="1000" rgba="0.9 0.2 0.1 1"/>')
                parts.append(f'<site name="{leg}_tip_site" pos="{_fmt(end)}" size="0.005"/>')
            closes.append("</body>")
        return "".join(parts) + "".join(reversed(closes))

    keys = []
    for key in LAYOUT:
        cx, cy = key_world_center(key)
        keys.append(
            f'<body name="key_{key.id}" pos="{cx:.4f} {cy:.4f} -0.05">'
            f'<joint name="keyjoint_{key.id}" type="slide" axis="0 0 1" range="{-KEY_TRAVEL} 0" limited="true" '
            f'stiffness="{params.key_stiffness}" springref="{params.key_preload}" damping="{params.key_damping}"/>'
            f'<geom name="keycap_{key.id}" type="box" size="{key.width / 2:.4f} {key.height / 2:.4f} 0.05" '
            f'contype="2" conaffinity="1" mass="{params.key_mass}" rgba="0.2 0.2 0.22 1"/></body>')
    actuators = [f'<position name="act_{j}" joint="{j}" kp="{params.kp * gain:.6g}" kv="{params.kv * gain:.6g}"/>'
                 for leg in LEGS for j in (f"joint_{leg}{dof}" for dof in fly.leg_dofs)]
    thorax_visual = (f'<geom type="ellipsoid" size="{0.6 * scale:.4f} {0.45 * scale:.4f} {0.4 * scale:.4f}" '
                     f'contype="0" conaffinity="0" density="0" rgba="0.5 0.3 0.15 1"/>')
    return f"""
<mujoco model="flyslop_keyboard">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{params.timestep}" gravity="0 0 0" integrator="implicitfast"/>
  <default><geom solref="{4 * params.timestep} 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    {deck_xml()}
    {''.join(keys)}
    <body name="thorax" mocap="true">{thorax_visual}{''.join(leg_xml(leg) for leg in LEGS)}</body>
  </worldbody>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>"""


def deck_rectangles(res: float = 0.05, border: float = 1.0) -> list[tuple[float, float, float, float]]:
    """Layout-frame rectangles (x0, y0, x1, y1) covering the deck but not the key holes.

    Key edges lie on a 0.05 grid, so a raster at that resolution is exact.
    Runs of free cells are merged per row, then identical runs across rows.
    """
    from .keyboard import hit_test
    nx, ny = round((LAYOUT_WIDTH + 2 * border) / res), round((LAYOUT_HEIGHT + 2 * border) / res)
    runs_by_row = []
    for j in range(ny):
        y = -border + (j + 0.5) * res
        runs, start = [], None
        for i in range(nx + 1):
            free = i < nx and hit_test(-border + (i + 0.5) * res, y) is None
            if free and start is None:
                start = i
            elif not free and start is not None:
                runs.append((start, i))
                start = None
        runs_by_row.append(runs)
    rects, open_runs = [], {}
    for j, runs in enumerate(runs_by_row + [[]]):
        current = set(runs)
        for run in list(open_runs):
            if run not in current:
                j0 = open_runs.pop(run)
                rects.append((-border + run[0] * res, -border + j0 * res, -border + run[1] * res, -border + j * res))
        for run in runs:
            open_runs.setdefault(run, j)
    return rects


def deck_xml() -> str:
    parts = []
    for x0, y0, x1, y1 in deck_rectangles():
        cx, cy = (x0 + x1) / 2 - LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2 - (y0 + y1) / 2
        parts.append(f'<geom type="box" pos="{cx:.4f} {cy:.4f} {-DECK_DEPTH - 0.5:.4f}" '
                     f'size="{(x1 - x0) / 2:.4f} {(y1 - y0) / 2:.4f} 0.5" contype="2" conaffinity="1" '
                     f'rgba="0.7 0.7 0.72 1"/>')
    # Palm rest in front of the keyboard, where the hind legs stand.
    x0, x1, y0, y1 = -1.2, LAYOUT_WIDTH + 1.2, LAYOUT_HEIGHT + 1.0, LAYOUT_HEIGHT + 10.5
    cx, cy = (x0 + x1) / 2 - LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2 - (y0 + y1) / 2
    parts.append(f'<geom type="box" pos="{cx:.4f} {cy:.4f} {-DECK_DEPTH - 0.5:.4f}" '
                 f'size="{(x1 - x0) / 2:.4f} {(y1 - y0) / 2:.4f} 0.5" contype="2" conaffinity="1" '
                 f'rgba="0.7 0.7 0.72 1"/>')
    return "".join(parts)


def _quat_mat(q) -> np.ndarray:
    from .flybody import quat_to_mat
    return quat_to_mat(q)


def fly_model_text() -> str:
    from .flybody import MODEL_PATH
    return MODEL_PATH.read_text()


class PhysicalKeyboard:
    """Emits ``contact_onset``/``key``/``contact_offset`` events from key travel plus claw contact.

    Event fields match :class:`backend.keyboard.ContactKeyboard`, so
    ``backend.embodiment.replay_text`` validates physics episodes too.
    """

    def __init__(self, sim: "FlySim"):
        self.sim = sim
        self.down: dict[str, str] = {}          # key_id -> foot holding it
        self.armed = np.ones(len(sim.key_ids), dtype=bool)
        # "latch" (one-shot Shift) keeps the scripted planner reproducible; "held" is contract C2's new mode.
        self.shift_mode = getattr(sim.params, "shift_mode", "latch")
        # "held" (default, contract C2): an Fn chord needs a foreleg on Fn while the other presses. "latch" is an
        # opt-in fallback for chords the forelegs cannot span (Fn is at the far left of the bottom row).
        self.fn_mode = getattr(sim.params, "fn_mode", "held")
        self.fn_latched = False
        self.contact_ids: dict[str, int] = {}   # key_id -> contact_id
        self.next_contact_id = 0
        self.shift_latched = False
        self.last_emission_tick = -10_000
        self.unattributed = 0

    def update(self, tick: int, depression: np.ndarray, touching: dict[int, str]) -> list[dict]:
        p = self.sim.params
        events: list[dict] = []
        for i in np.flatnonzero((~self.armed) & (depression < p.release_depth)):
            key_id = self.sim.key_ids[i]
            self.armed[i] = True
            foot = self.down.pop(key_id, None)
            if foot is not None:
                events.append({"tick": tick, "type": "contact_offset", "key_id": key_id, "foot": foot,
                               "contact_id": self.contact_ids.pop(key_id)})
                if key_id in MODIFIER_KEYS and (self.shift_mode == "held" or key_id == "Fn") \
                        and not (key_id == "Fn" and self.fn_mode == "latch"):
                    events.append({"tick": tick, "type": "modifier", "key_id": key_id, "foot": foot,
                                   "state": modifier_state(key_id, False)})
        for i in np.flatnonzero(self.armed & (depression >= p.press_depth)):
            key_id = self.sim.key_ids[i]
            self.armed[i] = False
            foot = touching.get(i)
            if foot is None or foot in self.down.values():
                self.unattributed += 1   # no claw on the cap (or claw busy): no character
                continue
            self.down[key_id] = foot
            contact_id = self.contact_ids[key_id] = self.next_contact_id
            self.next_contact_id += 1
            events.append({"tick": tick, "type": "contact_onset", "key_id": key_id, "foot": foot,
                           "contact_id": contact_id})
            held_mode = self.shift_mode == "held"
            if key_id == "Fn" and self.fn_mode == "latch":
                self.fn_latched = True
                events.append({"tick": tick, "type": "modifier", "key_id": key_id, "foot": foot, "state": "fn_latched"})
            elif key_id in MODIFIER_KEYS and (held_mode or key_id == "Fn"):
                events.append({"tick": tick, "type": "modifier", "key_id": key_id, "foot": foot,
                               "state": modifier_state(key_id, True)})
            if tick - self.last_emission_tick < p.debounce_ticks:
                continue
            if not held_mode and key_id in SHIFT_KEYS:
                self.shift_latched = True
                events.append({"tick": tick, "type": "modifier", "key_id": key_id, "foot": foot,
                               "state": "shift_latched"})
            latched = not held_mode and self.shift_latched
            mods = [m for m in MODIFIER_KEYS if m in self.down and m != key_id]
            if latched:
                mods = [m for m in mods if m not in SHIFT_KEYS] + ["ShiftLeft"]
            fn_latched = self.fn_latched and key_id != "Fn"
            if fn_latched and "Fn" not in mods:
                mods.append("Fn")
            character = key_char(key_id, mods)
            if emits_key(key_id, character):
                event = {"tick": tick, "type": "key", "key_id": key_id, "key": key_id, "foot": foot,
                         "modifiers": mods, "char": character, "contact_id": contact_id}
                if latched:
                    event["shift_latched"] = True   # a tapped, not held, modifier: replay_text skips the held-at-onset check
                if fn_latched:
                    event["fn_latched"] = True      # same for the sticky-Fn latch (fn_mode="latch")
                events.append(event)
                self.last_emission_tick = tick
                self.shift_latched = False
                if fn_latched:
                    self.fn_latched = False
        return events

    def holding(self, foot: str) -> str | None:
        return next((k for k, f in self.down.items() if f == foot), None)


class FlySim:
    """Tick-level interface: carry the thorax, command leg joints, read contacts and keys."""

    def __init__(self, params: PhysicsParams | None = None, scale: float | None = None):
        from .embodiment import MM
        self.params = params or PhysicsParams()
        self.scale = MM if scale is None else scale
        self.fly = default_body()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.params, self.scale))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.key_ids = [key.id for key in LAYOUT]
        self.key_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"keyjoint_{k}")]
                                  for k in self.key_ids])
        self.keycap_geom = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"keycap_{k}"): i
                            for i, k in enumerate(self.key_ids)}
        self.tip_geom = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_tip"): leg for leg in LEGS}
        self.tip_site = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_tip_site") for leg in LEGS])
        self.leg_qadr = np.array([[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{dof}")]
                                   for dof in self.fly.leg_dofs] for leg in LEGS])
        self.leg_dadr = np.array([[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{dof}")]
                                   for dof in self.fly.leg_dofs] for leg in LEGS])
        self.tick = 0
        self.substep = 0
        self.last_touch: dict[int, tuple[str, int]] = {}
        self.keyboard = PhysicalKeyboard(self)

    # -- state -----------------------------------------------------------
    def reset(self, body: np.ndarray, q: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.set_body(body)
        self.data.qpos[self.leg_qadr.ravel()] = q.ravel()
        self.data.ctrl[:] = q.ravel()
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.substep = 0
        self.last_touch = {}
        self.keyboard = PhysicalKeyboard(self)

    def set_body(self, body: np.ndarray) -> None:
        """Thorax pose ``(x, y, z, yaw, pitch, roll)`` in world key pitches."""
        self.data.mocap_pos[0] = body[:3]
        self.data.mocap_quat[0] = _mat_to_quat(euler_zyx(body[3], body[4], body[5]))

    def tips(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site].copy()

    def joint_q(self) -> np.ndarray:
        return self.data.qpos[self.leg_qadr].copy()

    def depression(self) -> np.ndarray:
        return -self.data.qpos[self.key_qadr]

    def touching(self) -> dict[int, str]:
        """Key index -> leg whose claw is in contact with that cap."""
        out: dict[int, str] = {}
        for contact in self.data.contact[: self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            for tip, cap in ((g1, g2), (g2, g1)):
                if tip in self.tip_geom and cap in self.keycap_geom:
                    out[self.keycap_geom[cap]] = self.tip_geom[tip]
        return out

    # -- stepping --------------------------------------------------------
    def step(self, q_target: np.ndarray, body: np.ndarray | None = None) -> list[dict]:
        """Advance one tick; returns keyboard events emitted during it."""
        self.tick += 1
        if body is not None:
            self.set_body(body)
        start, goal = self.data.ctrl.copy(), np.asarray(q_target, dtype=float).ravel()
        events: list[dict] = []
        window = max(1, round(self.params.touch_window / self.params.timestep))
        n = self.params.substeps
        for k in range(n):
            # Joint targets ramp across the tick instead of stepping once per tick.
            self.data.ctrl[:] = start + (goal - start) * ((k + 1) / n)
            mujoco.mj_step(self.model, self.data)
            self.substep += 1
            for key_index, leg in self.touching().items():
                self.last_touch[key_index] = (leg, self.substep)
            depression = self.depression()
            if (depression >= self.params.press_depth).any() or (~self.keyboard.armed).any():
                recent = {k: leg for k, (leg, when) in self.last_touch.items() if self.substep - when <= window}
                events += self.keyboard.update(self.tick, depression, recent)
        return events

    # -- targets ---------------------------------------------------------
    def solve_targets(self, body: np.ndarray, targets_world: np.ndarray) -> tuple[np.ndarray, float]:
        """IK for six claw targets in world coordinates -> joint targets (6, 7)."""
        return self.solve_many(body[None], targets_world[None])
        
    def solve_many(self, bodies: np.ndarray, targets_world: np.ndarray) -> tuple[np.ndarray, float]:
        """Batched IK over ticks: (T, 6) bodies, (T, 6, 3) claw targets -> (T, 6, 7) joints."""
        t = len(bodies)
        rot = np.array([euler_zyx(b[3], b[4], b[5]) for b in bodies])
        local = np.einsum("tji,tlj->tli", rot, targets_world - bodies[:, None, :3]) / self.scale
        q, error = self.fly.solve_batch(np.tile(np.arange(6), t), local.reshape(-1, 3))
        q = q.reshape(t, 6, -1)
        return (q[0] if t == 1 else q), float(error.max()) * self.scale


class LegIK:
    """Fast warm-started IK for a few legs using MuJoCo kinematics and site Jacobians.

    Works in the thorax frame of a separate ``MjData`` so the simulation state
    is untouched. Used per control step by the learning environment.
    """

    def __init__(self, sim: "FlySim", legs=LEGS, damping: float = 1e-4, posture: float = 2e-3):
        self.sim, self.legs = sim, list(legs)
        self.data = mujoco.MjData(sim.model)
        self.index = [LEGS.index(leg) for leg in self.legs]
        self.damping, self.posture = damping, posture
        self.rest = np.array([sim.fly.rest_q[leg] for leg in LEGS])
        self.jacp = np.zeros((3, sim.model.nv))

    def solve(self, body: np.ndarray, targets_world: np.ndarray, q0: np.ndarray, iterations: int = 4) -> np.ndarray:
        """Joint targets (6, 7): legs in ``self.legs`` solved toward ``targets_world`` rows, others kept."""
        m, d = self.sim.model, self.data
        d.mocap_pos[0] = body[:3]
        d.mocap_quat[0] = _mat_to_quat(euler_zyx(body[3], body[4], body[5]))
        q = q0.copy()
        eye = np.eye(q.shape[1])
        for _ in range(iterations):
            d.qpos[self.sim.leg_qadr.ravel()] = q.ravel()
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)   # site Jacobians need cdof
            for leg in self.index:
                site = self.sim.tip_site[leg]
                err = targets_world[leg] - d.site_xpos[site]
                mujoco.mj_jacSite(m, d, self.jacp, None, site)
                jac = self.jacp[:, self.sim.leg_dadr[leg]]
                rhs = jac.T @ err + self.posture * (self.rest[leg] - q[leg])
                step = np.linalg.solve(jac.T @ jac + (self.damping + self.posture) * eye, rhs)
                q[leg] = q[leg] + np.clip(step, -0.3, 0.3)
        return q

    def tips(self, body: np.ndarray, q: np.ndarray) -> np.ndarray:
        d = self.data
        d.mocap_pos[0] = body[:3]
        d.mocap_quat[0] = _mat_to_quat(euler_zyx(body[3], body[4], body[5]))
        d.qpos[self.sim.leg_qadr.ravel()] = q.ravel()
        mujoco.mj_kinematics(self.sim.model, d)
        return d.site_xpos[self.sim.tip_site].copy()


def support_height(x: float, y: float, radius: float) -> float:
    """Highest surface under a claw footprint (key top 0, deck -DECK_DEPTH)."""
    from .embodiment import ground
    samples = [(0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)]
    return max(ground(x + dx, y + dy) for dx, dy in samples)


def physical_targets(targets: np.ndarray, radius: float, push_through: float = 0.06) -> np.ndarray:
    """Planner claw targets -> physics claw-centre targets.

    Stance claws (target on the surface) rest on the highest surface under the
    claw sphere. Targets below the key top are strikes and are pushed through
    so the actuator, not the target, sets the contact force.
    """
    from .embodiment import ground
    out = targets.copy()
    for i, (x, y, z) in enumerate(targets):
        if abs(z - ground(x, y)) < 1e-9:
            out[i, 2] = support_height(x, y, radius) + radius
        elif z < 0.0:
            out[i, 2] = z - push_through + radius
        else:
            out[i, 2] = z + radius
    return out


def physics_episode(target_id: str, target: str, params: PhysicsParams | None = None) -> dict:
    """Scripted planner through MuJoCo: the same gait/press plan as the kinematic demo,
    but key output comes only from physically depressed, claw-touched keys."""
    from .embodiment import Planner, key_sequence, synthetic_activity, text_hash, to_layout, wrap

    planner = Planner()
    planner.idle(4)
    planner.type_keys(key_sequence(target))
    if planner.hover_leg:
        planner.lower_leg(planner.hover_leg)
    planner.idle(6)
    sim = FlySim(params)
    bodies = np.array(planner.bodies)
    goals = np.array([physical_targets(t, sim.params.tip_radius, sim.params.push_through) for t in planner.targets])
    q_targets, max_ik = sim.solve_many(bodies, goals)
    sim.reset(bodies[0], q_targets[0])
    result = run_physics(sim, target_id, target, bodies, goals, planner.phase,
                         mode="scripted_physics_neuromechfly", q_targets=q_targets)
    result["metadata"]["max_ik_error"] = round(max_ik, 5)
    return result


def run_physics(sim: FlySim, target_id: str, target: str, bodies: np.ndarray, goals: np.ndarray,
                phases: list[str], mode: str, q_targets: np.ndarray | None = None) -> dict:
    """Step a planned body/claw-target sequence through physics and package a replay."""
    from .editor import Editor
    from .embodiment import FLY_SCALE, annotate_key_event, synthetic_activity, text_hash, to_layout, wrap
    from .keyboard import KEY_PITCH_MM
    editor, events, rows, max_ik = Editor(), [], [], 0.0
    if q_targets is None:
        q_targets, max_ik = sim.solve_many(bodies, goals)
    for tick in range(len(bodies)):
        body = bodies[tick]
        q = q_targets[tick]
        emitted = sim.step(q, body) if tick else []
        tips = sim.tips()
        body_pose = {k: round(float(v), 3) for k, v in zip(("x", "y", "z", "yaw"), body[:4])}
        for event in emitted:
            leg = LEGS.index(event["foot"])
            event["body"] = body_pose
            event["foot_pose"] = {a: round(float(v), 4) for a, v in zip("xyz", tips[leg])}
            if event["type"] == "key":
                annotate_key_event(editor, event)
            events.append(event)
        previous = bodies[max(0, tick - 1)]
        activity = synthetic_activity(tips, phases[tick], float(np.linalg.norm(body[:2] - previous[:2])),
                                      wrap(body[3] - previous[3]))
        rows.append([tick, *np.round(body, 4).tolist(), *np.round(sim.joint_q().ravel(), 3).tolist(),
                     *np.round(tips.ravel(), 3).tolist(), *activity])
    fields = ["tick", "x", "y", "z", "yaw", "pitch", "roll"]
    fields += [f"{leg}.{dof}" for leg in LEGS for dof in sim.fly.leg_dofs]
    fields += [f"{leg}.tip.{axis}" for leg in LEGS for axis in "xyz"]
    fields += [f"activity.{i}" for i in range(16)]
    p = sim.params
    return {
        "metadata": {
            "target_id": target_id, "mode": mode, "observation_condition": "state_assisted", "seed": 0,
            "tick_hz": p.tick_hz,
            "body_model": "NeuroMechFly v2 tree in MuJoCo: 42 position-actuated leg joints, claw-sphere contacts; "
                          "thorax carried by mocap, gravity off",
            "physics": {"timestep": p.timestep, "substeps": p.substeps, "kp": p.kp, "kv": p.kv,
                        "key_stiffness": p.key_stiffness, "press_depth": p.press_depth,
                        "release_depth": p.release_depth, "tip_radius": p.tip_radius,
                        "mujoco": mujoco.__version__},
            "fly_scale": FLY_SCALE, "key_pitch_mm": KEY_PITCH_MM, "frame": "X right, Y toward screen, Z up; key pitch units; origin keyboard centre",
            "contact_threshold_z": -p.press_depth, "max_ik_error": round(max_ik, 5),
            "unattributed_presses": sim.keyboard.unattributed,
        },
        "frames": {"fields": fields, "rows": rows},
        "events": events, "target": target, "final_text": editor.text, "final_text_hash": text_hash(editor.text),
    }
