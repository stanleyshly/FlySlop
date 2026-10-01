"""Contact-gated, kinematic Gymnasium environment for keyboard reaches.

This is a state-assisted point-foot scaffold. It is not a physical fly model,
and it does not imply learned behavior. The action directly sets a foot pose;
only :class:`ContactKeyboard` can emit characters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # optional extra
    raise ImportError("GymnasiumEnv requires the optional 'gymnasium' dependency") from exc

from .embodiment import apply_key, text_hash
from .keyboard import BY_ID, CHAR_TO_KEY, ContactKeyboard, hit_test


DEFAULT_TARGETS = ("fly", "abc", "sv", "module")


class KeyboardReachEnv(gym.Env):
    """Move one point foot to type a target using real contact state transitions.

    Action is an absolute ``(x, y, z)`` foot position in the scaled keyboard
    coordinates documented by ``backend.keyboard``. Observation includes the
    next key center, making this a deliberately state-assisted task.
    """

    metadata = {"render_modes": []}

    def __init__(self, targets: tuple[str, ...] = DEFAULT_TARGETS, max_steps: int = 500):
        super().__init__()
        if not targets or any(not target for target in targets):
            raise ValueError("targets must contain non-empty strings")
        unsupported = sorted({char for target in targets for char in target if char not in CHAR_TO_KEY})
        if unsupported:
            raise ValueError(f"Unsupported target characters: {unsupported}")
        self.targets = tuple(targets)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -0.2], dtype=np.float32),
            high=np.array([16.0, 5.0, 0.8], dtype=np.float32), dtype=np.float32,
        )
        # foot xyz, next key xy, shifted bit, progress, most recent success bit
        self.observation_space = spaces.Box(
            low=np.array([0, -0.5, -0.2, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([16, 5, 0.8, 16, 5, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.keyboard = ContactKeyboard()
        self.target = ""
        self.buffer = ""
        self.tick = 0
        self.foot = np.array([0.35, -0.3, 0.6], dtype=np.float32)
        self.last_success = 0.0
        self.events: list[dict[str, Any]] = []

    def _observation(self) -> np.ndarray:
        index = min(len(self.buffer), len(self.target) - 1)
        key_id, shifted = CHAR_TO_KEY[self.target[index]]
        key = BY_ID[key_id]
        return np.asarray([
            *self.foot, key.x + key.width / 2, key.y + key.height / 2,
            float(shifted), len(self.buffer) / len(self.target), self.last_success,
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        target = options.get("target")
        if target is None:
            target = self.targets[int(self.np_random.integers(len(self.targets)))]
        if not isinstance(target, str) or not target or any(c not in CHAR_TO_KEY for c in target):
            raise ValueError("target must be a non-empty string using supported keyboard characters")
        self.target = target
        self.buffer = ""
        self.tick = 0
        self.foot = np.array([0.35, -0.3, 0.6], dtype=np.float32)
        self.keyboard = ContactKeyboard()
        self.events = []
        self.last_success = 0.0
        return self._observation(), {"target": self.target, "mode": "state_assisted_kinematic"}

    def step(self, action):
        pose = np.asarray(action, dtype=np.float32)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("action must be a finite (x, y, z) vector")
        pose = np.clip(pose, self.action_space.low, self.action_space.high)
        self.foot = pose
        self.tick += 1
        # Shift must be actuated through its own key contact. The observation
        # says whether the next target needs Shift, but cannot emit it.
        emitted = self.keyboard.update(self.tick, float(pose[0]), float(pose[1]), float(pose[2]))
        reward = -0.01
        self.last_success = 0.0
        for event in emitted:
            self.events.append(event.copy())
            if event["type"] != "key":
                continue
            expected = self.target[len(self.buffer)] if len(self.buffer) < len(self.target) else None
            self.buffer += event["char"]
            correct = event["char"] == expected
            reward += 1.0 if correct else -1.0
            self.last_success = float(correct)
            event["text"] = self.buffer
            event["expected"] = expected
        terminated = self.buffer == self.target
        truncated = self.tick >= self.max_steps and not terminated
        if terminated:
            reward += 2.0
        return self._observation(), reward, terminated, truncated, {
            "target": self.target,
            "text": self.buffer,
            "events": emitted,
            "tick": self.tick,
            "exact": terminated,
        }


def physical_key_for(char: str, shift_latched: bool) -> str:
    """Next key a foot must press to produce ``char`` (Shift first if needed)."""
    key_id, needs_shift = CHAR_TO_KEY[char]
    return "ShiftLeft" if needs_shift and not shift_latched else key_id


def key_center(key_id: str) -> tuple[float, float]:
    key = BY_ID[key_id]
    return key.x + key.width / 2, key.y + key.height / 2


class KeyboardTypingEnv(gym.Env):
    """PPO-oriented variant: relative foot motion, dense shaping, contact-gated output.

    Action is a normalized ``(dx, dy, dz)`` in ``[-1, 1]`` scaled by
    ``max_xy_step``/``max_z_step`` per tick. Output still only comes from
    :class:`ContactKeyboard` press/release transitions, so the policy must
    learn travel, press, and lift-to-rearm. The observation is state assisted:
    it contains the offset to the next *physical* key (Shift before a shifted
    character). A wrong character ends the episode when
    ``terminate_on_error`` is set, which keeps the buffer a target prefix.

    Reward (frozen via ``reward`` weights): ``correct`` per right character,
    ``wrong`` per wrong character, ``shift`` for a needed Shift press and
    ``-shift`` for an unneeded one, ``complete`` at the end, ``time`` every
    tick, plus potential shaping ``shaping * (phi' - phi)`` (phi' = 0 on success) where
    ``phi`` rewards being over the target key and lowered when armed or
    raised when not armed.
    """

    metadata = {"render_modes": []}
    OBS_LABELS = ("foot_x", "foot_y", "foot_z", "dx", "dy", "over_target", "armed",
                  "shift_needed", "shift_latched", "progress", "time_left", "last_correct")
    DEFAULT_REWARD = {"correct": 1.0, "wrong": -1.0, "shift": 0.5, "complete": 2.0,
                      "time": -0.005, "shaping": 1.0}
    LOW = np.array([0.0, -0.5, -0.2], dtype=np.float32)
    HIGH = np.array([16.0, 5.6, 0.8], dtype=np.float32)
    REST = np.array([7.5, 2.6, 0.6], dtype=np.float32)

    def __init__(self, targets: tuple[str, ...] = DEFAULT_TARGETS, steps_per_char: int = 40,
                 max_xy_step: float = 0.6, max_z_step: float = 0.3, terminate_on_error: bool = True,
                 random_start: bool = True, gamma: float = 0.99, reward: dict | None = None):
        super().__init__()
        self.set_targets(targets)
        self.steps_per_char = int(steps_per_char)
        self.step_scale = np.array([max_xy_step, max_xy_step, max_z_step], dtype=np.float32)
        self.terminate_on_error = bool(terminate_on_error)
        self.random_start = bool(random_start)
        self.gamma = float(gamma)
        self.reward_weights = {**self.DEFAULT_REWARD, **(reward or {})}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(len(self.OBS_LABELS),), dtype=np.float32)
        self.keyboard = ContactKeyboard()
        # ``buffer`` is the correct target prefix; ``typed`` is the authoritative
        # editor text including any wrong final character.
        self.target, self.buffer, self.typed, self.tick, self.max_steps = "", "", "", 0, 1
        self.foot = self.REST.copy()
        self.last_correct = 0.0
        self.events: list[dict[str, Any]] = []
        self.poses: list[list[float]] = []

    def set_targets(self, targets) -> None:
        targets = tuple(targets)
        if not targets or any(not target for target in targets):
            raise ValueError("targets must contain non-empty strings")
        unsupported = sorted({c for t in targets for c in t if c not in CHAR_TO_KEY})
        if unsupported:
            raise ValueError(f"Unsupported target characters: {unsupported}")
        self.targets = targets

    # -- state helpers -------------------------------------------------
    def _armed(self) -> bool:
        return self.keyboard.armed.get("right_foreleg", True) and "right_foreleg" not in self.keyboard.down

    def _next_key(self) -> tuple[str | None, bool]:
        if len(self.buffer) >= len(self.target):
            return None, False
        char = self.target[len(self.buffer)]
        return physical_key_for(char, self.keyboard.shift_latched), CHAR_TO_KEY[char][1]

    def _geometry(self) -> tuple[float, float, bool]:
        key_id, _ = self._next_key()
        if key_id is None:
            return 0.0, 0.0, False
        cx, cy = key_center(key_id)
        over = hit_test(float(self.foot[0]), float(self.foot[1])) == key_id
        return cx - float(self.foot[0]), cy - float(self.foot[1]), over

    def _potential(self) -> float:
        """Closer to the next key is better; over it, lower when armed, lift when not."""
        dx, dy, over = self._geometry()
        phi = -0.2 * float(np.hypot(dx, dy))
        if over:
            z = float(self.foot[2])
            phi += 0.5 * ((0.8 - z) if self._armed() else (z + 0.2))
        return phi

    def _observation(self) -> np.ndarray:
        dx, dy, over = self._geometry()
        _key, shift_needed = self._next_key()
        span = self.HIGH - self.LOW
        foot = 2.0 * (self.foot - self.LOW) / span - 1.0
        obs = np.array([
            *foot, np.clip(dx / 8.0, -1, 1), np.clip(dy / 3.0, -1, 1), float(over), float(self._armed()),
            float(shift_needed), float(self.keyboard.shift_latched),
            len(self.buffer) / len(self.target), 1.0 - self.tick / self.max_steps, self.last_correct,
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    # -- gym API -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        target = options.get("target")
        if target is None:
            target = self.targets[int(self.np_random.integers(len(self.targets)))]
        if not isinstance(target, str) or not target or any(c not in CHAR_TO_KEY for c in target):
            raise ValueError("target must be a non-empty string using supported keyboard characters")
        self.target, self.buffer, self.typed, self.tick = target, "", "", 0
        self.max_steps = self.steps_per_char * len(target)
        if self.random_start:
            self.foot = np.array([self.np_random.uniform(0.5, 14.5), self.np_random.uniform(0.8, 5.2),
                                  self.np_random.uniform(0.3, 0.8)], dtype=np.float32)
        else:
            self.foot = self.REST.copy()
        self.keyboard = ContactKeyboard()
        self.events, self.poses = [], [self.foot.tolist()]
        self.last_correct = 0.0
        return self._observation(), {"target": self.target, "mode": "state_assisted_kinematic_relative"}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite 3-vector")
        w = self.reward_weights
        phi = self._potential()
        shift_needed_before = self._next_key()[1] and not self.keyboard.shift_latched
        self.foot = np.clip(self.foot + np.clip(action, -1, 1) * self.step_scale, self.LOW, self.HIGH)
        self.tick += 1
        self.poses.append(self.foot.tolist())
        emitted = self.keyboard.update(self.tick, float(self.foot[0]), float(self.foot[1]), float(self.foot[2]))
        reward = w["time"]
        self.last_correct = 0.0
        error = False
        for event in emitted:
            if event["type"] == "key":
                self.typed = apply_key(self.typed, event["key_id"], event["char"])
                event = {**event, "text": self.typed, "text_hash": text_hash(self.typed)}
            self.events.append(event)
            if event["type"] == "modifier":
                reward += w["shift"] if shift_needed_before else -w["shift"]
            if event["type"] != "key":
                continue
            expected = self.target[len(self.buffer)] if len(self.buffer) < len(self.target) else None
            correct = event["char"] == expected and event["key_id"] != "Backspace"
            if correct:
                self.buffer += event["char"]
                reward += w["correct"]
                self.last_correct = 1.0
            else:
                reward += w["wrong"]
                error = True
        if error and not self.terminate_on_error:
            self.buffer = self.typed
        exact = self.typed == self.target
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
        return self._observation(), float(reward), terminated, truncated, {
            "target": self.target, "text": self.typed, "events": emitted, "tick": self.tick,
            "exact": exact, "error": error,
        }
