"""NeuroMechFly v2 leg kinematics without a MuJoCo runtime dependency.

The kinematic tree, joint axes, and rest posture are exported from FlyGym by
``scripts/export_neuromechfly.py``; that script checks this forward
kinematics against MuJoCo. Units are NeuroMechFly millimetres in the thorax
frame (x forward, y left, z up). Inverse kinematics is damped least squares
toward the recorded neutral walking posture. This is kinematic posing, not a
physics simulation: there are no joint torques, masses, or ground forces.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "neuromechfly" / "model.json"
SEGMENTS = ["Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return axis_angle(np.array([0.0, 0, 1]), yaw) @ axis_angle(np.array([0.0, 1, 0]), pitch) @ axis_angle(np.array([1.0, 0, 0]), roll)


class FlyBody:
    def __init__(self, path: Path = MODEL_PATH):
        model = json.loads(path.read_text())
        self.legs: list[str] = model["legs"]
        self.leg_dofs: list[str] = model["leg_dofs"]
        bodies = {body["name"]: body for body in model["bodies"]}
        self.rest = {j["name"]: j["rest"] for body in model["bodies"] for j in body["joints"]}
        self.tip_offset = {leg: np.asarray(v) for leg, v in model["tips"].items()}
        self.chains: dict[str, list[tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]]] = {}
        for leg in self.legs:
            chain = []
            for segment in SEGMENTS:
                body = bodies[f"{leg}{segment}"]
                if any(any(j["pos"]) for j in body["joints"]):
                    raise ValueError("Offset joint anchors are not supported")
                joints = [(j["name"], np.asarray(j["axis"], dtype=float)) for j in body["joints"]]
                chain.append((np.asarray(body["pos"], dtype=float), quat_to_mat(body["quat"]), joints))
            self.chains[leg] = chain
        self.rest_q = {leg: np.array([self.rest[f"joint_{leg}{dof}"] for dof in self.leg_dofs]) for leg in self.legs}

    def leg_chain_positions(self, leg: str, angles: dict[str, float]) -> list[np.ndarray]:
        """Body origins of the leg chain in the thorax frame for named joint angles."""
        rot, pos, out = np.eye(3), np.zeros(3), []
        for offset, local, joints in self.chains[leg]:
            pos = pos + rot @ offset
            rot = rot @ local
            for joint_name, axis in joints:
                rot = rot @ axis_angle(axis, angles.get(joint_name, self.rest[joint_name]))
            out.append(pos)
        return out

    def _dof_angles(self, leg: str, q: np.ndarray) -> dict[str, float]:
        return {f"joint_{leg}{dof}": float(v) for dof, v in zip(self.leg_dofs, q)}

    def tip_and_jacobian(self, leg: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Claw tip in the thorax frame and its Jacobian for the leg DoFs."""
        angles = self._dof_angles(leg, q)
        rot, pos = np.eye(3), np.zeros(3)
        anchors: list[tuple[np.ndarray, np.ndarray]] = []
        for offset, local, joints in self.chains[leg]:
            pos = pos + rot @ offset
            rot = rot @ local
            for joint_name, axis in joints:
                if joint_name in angles:
                    anchors.append((pos, rot @ axis))
                rot = rot @ axis_angle(axis, angles.get(joint_name, self.rest[joint_name]))
        tip = pos + rot @ self.tip_offset[leg]
        jac = np.stack([np.cross(axis, tip - anchor) for anchor, axis in anchors], axis=1)
        return tip, jac

    def tip(self, leg: str, q: np.ndarray) -> np.ndarray:
        return self.tip_and_jacobian(leg, q)[0]

    def solve_leg(self, leg: str, target: np.ndarray, q0: np.ndarray | None = None,
                  iterations: int = 12, tolerance: float = 2e-3) -> tuple[np.ndarray, float]:
        """Damped least-squares IK with a pull toward the neutral walking posture."""
        rest = self.rest_q[leg]
        q = rest.copy() if q0 is None else q0.copy()
        damping, posture = 2e-3, 4e-3
        eye = np.eye(len(q))
        error = np.inf
        for _ in range(iterations):
            tip, jac = self.tip_and_jacobian(leg, q)
            delta = target - tip
            error = float(np.linalg.norm(delta))
            if error < tolerance:
                break
            step = np.linalg.solve(jac.T @ jac + (damping + posture) * eye, jac.T @ delta + posture * (rest - q))
            q = q + np.clip(step, -0.35, 0.35)
        else:
            error = float(np.linalg.norm(target - self.tip(leg, q)))
        return q, error


    def _batch_arrays(self):
        """Stack the six chains; they share joint structure (asserted)."""
        if hasattr(self, "_batch"):
            return self._batch
        layout = [[len(joints) for _, _, joints in self.chains[leg]] for leg in self.legs]
        if any(row != layout[0] for row in layout):
            raise ValueError("Leg chains differ in joint structure")
        offsets = np.array([[off for off, _, _ in self.chains[leg]] for leg in self.legs])
        locals_ = np.array([[rot for _, rot, _ in self.chains[leg]] for leg in self.legs])
        axes = [np.array([[axis for _, axis in self.chains[leg][b][2]] for leg in self.legs]) for b in range(len(SEGMENTS))]
        first = self.legs[0]
        dof_names = {f"joint_{first}{dof}" for dof in self.leg_dofs}
        fixed = [np.array([[self.rest[name] for name, _ in self.chains[leg][b][2]] for leg in self.legs]) for b in range(len(SEGMENTS))]
        is_dof = [[name in dof_names for name, _ in self.chains[first][b][2]] for b in range(len(SEGMENTS))]
        tips = np.array([self.tip_offset[leg] for leg in self.legs])
        rest = np.array([self.rest_q[leg] for leg in self.legs])
        self._batch = offsets, locals_, axes, fixed, is_dof, tips, rest
        return self._batch

    def batch_tip_and_jacobian(self, leg_index: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        offsets, locals_, axes, fixed, is_dof, tips, _ = self._batch_arrays()
        n = len(leg_index)
        rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        pos = np.zeros((n, 3))
        anchors, world_axes = [], []
        k = 0
        for b in range(len(SEGMENTS)):
            pos = pos + np.einsum("nij,nj->ni", rot, offsets[leg_index, b])
            rot = rot @ locals_[leg_index, b]
            for j, dof in enumerate(is_dof[b]):
                axis = axes[b][leg_index, j]
                angle = q[:, k] if dof else fixed[b][leg_index, j]
                if dof:
                    anchors.append(pos)
                    world_axes.append(np.einsum("nij,nj->ni", rot, axis))
                    k += 1
                rot = rot @ batch_axis_angle(axis, angle)
        tip = pos + np.einsum("nij,nj->ni", rot, tips[leg_index])
        jac = np.stack([np.cross(a, tip - p) for p, a in zip(anchors, world_axes)], axis=2)
        return tip, jac

    def solve_batch(self, leg_index: np.ndarray, targets: np.ndarray, iterations: int = 18) -> tuple[np.ndarray, np.ndarray]:
        """Batched IK: joint angles (N, 7) and residual tip error (N,) in mm."""
        rest = self._batch_arrays()[6][leg_index]
        q = rest.copy()
        eye = np.eye(q.shape[1])
        for it in range(iterations):
            # Posture pull shapes the solution early, then drops out for accuracy.
            posture = 4e-3 if it < iterations - 4 else 0.0
            tip, jac = self.batch_tip_and_jacobian(leg_index, q)
            jt = np.transpose(jac, (0, 2, 1))
            rhs = np.einsum("nij,nj->ni", jt, targets - tip) + posture * (rest - q)
            step = np.linalg.solve(jt @ jac + (2e-3 + posture) * eye, rhs[..., None])[..., 0]
            q = q + np.clip(step, -0.35, 0.35)
        tip, _ = self.batch_tip_and_jacobian(leg_index, q)
        return q, np.linalg.norm(targets - tip, axis=1)


def batch_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    return np.stack([
        np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], axis=1),
        np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], axis=1),
        np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], axis=1),
    ], axis=1)


@lru_cache(maxsize=1)
def default_body() -> FlyBody:
    return FlyBody()
