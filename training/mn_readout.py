"""MN-driven foreleg readout (PLAN §2 "Typing connectome / Outputs", work package P3).

There is no free linear map from network activity to the leg joints. Each of the 14 foreleg joint targets is::

    target_j = rest_j + g_j * (sum of agonist MN rates - sum of antagonist MN rates) + b_j

with the sums over the side-matched motor neurons named by ``data/connectome/mn_joint_map.json`` (RF joints use the
right-side MNs, LF joints the left-side MNs) and only ``g_j`` and ``b_j`` learned: 28 parameters. In the env's
normalised action units ``target_j = rest_j + a_j * JOINT_SCALE[dof]``, so the readout outputs ``a_j = g_j * drive_j + b_j``
and the env applies the rest pose and scale.

* One-sided joints (no antagonist MN type in the map, currently Femur_roll) use the agonist sum only and are
  listed in ``readout.one_sided``.
* Joints with map ``status == "fallback"`` (none at present) would use a labelled small DN-rate linear readout
  (``readout.fallback_joints``); they are not MN driven.
* The thorax velocity (2 outputs) is **not MN driven**: it is a small linear readout of the DN rates
  (``readout.velocity_head``, ``readout.body_velocity_source == "DN linear readout (not MN driven)"``). MN driven
  walking with T2/T3 MNs is a later extension.

The readout registers as ``mn_antagonist``. Its output is the whole 16-D env action, so the policy replaces SB3's
linear ``action_net`` by the identity (``identity_action = True``, see ``ConnectomeActorCriticPolicy._build``).

The sign of ``g_j`` is initialised from ``data/connectome/mn_joint_sign_check.json`` (written by
``python -m training.mn_readout --verify-signs``), else +1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.fly_env import JOINT_DOFS, JOINT_NAMES, TYPERS
from training.connectome_policy import Readout, register_readout

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "data/connectome/mn_joint_map.json"
SIGN_PATH = ROOT / "data/connectome/mn_joint_sign_check.json"
MN_ACTION_DIM = 2 + len(JOINT_NAMES)


class MNAntagonistReadout(Readout):
    identity_action = True

    def __init__(self, runtime, map_path=MAP_PATH, sign_path=SIGN_PATH, arrays: dict | None = None):
        super().__init__()
        arrays = arrays if arrays is not None else getattr(runtime, "circuit_arrays", None)
        if arrays is None or "mn_type" not in arrays:
            from training.connectome_policy import DEFAULT_CIRCUIT, resolve_circuit
            arrays, _ = resolve_circuit(DEFAULT_CIRCUIT)
        if "mn_type" not in arrays:
            raise ValueError("mn_antagonist readout needs a circuit with mn_type/mn_side (typing_circuit.npz)")
        jmap = json.loads(Path(map_path).read_text())["joints"]
        signs = {}
        if Path(sign_path).exists():
            signs = {k: v["agonist_gain_sign"] for k, v in json.loads(Path(sign_path).read_text())["joints"].items()}
        idx_mn = np.asarray(arrays["idx_MN"], np.int64)
        pos = {int(n): i for i, n in enumerate(idx_mn)}
        mn_type, mn_side = arrays["mn_type"], arrays["mn_side"]
        n_dn = int(runtime.idx_DN.numel())
        coeff = np.zeros((len(JOINT_NAMES), len(idx_mn)), np.float32)   # +1 agonist, -1 antagonist
        g0 = np.ones(len(JOINT_NAMES), np.float32)
        self.joint_names = list(JOINT_NAMES)
        self.one_sided, self.fallback_joints, self.counts = [], [], {}
        for j, name in enumerate(JOINT_NAMES):
            spec, side = jmap[name], jmap[name]["side"]
            ag = [pos[int(n)] for n in idx_mn if mn_type[n] in spec["agonist"] and mn_side[n] == side]
            an = [pos[int(n)] for n in idx_mn if mn_type[n] in spec["antagonist"] and mn_side[n] == side]
            if spec["status"] == "fallback":
                self.fallback_joints.append(name)
            elif not ag:
                raise ValueError(f"{name}: no agonist MN in the circuit")
            if not spec["antagonist"]:
                self.one_sided.append(name)
            coeff[j, ag], coeff[j, an] = 1.0, -1.0
            self.counts[name] = (len(ag), len(an))
            g0[j] = np.sign(signs.get(name, 1) or 1) / np.sqrt(max(1, len(ag) + len(an)))
        self.register_buffer("coeff", torch.as_tensor(coeff))
        self.gain = nn.Parameter(torch.as_tensor(g0))               # g_j (14)
        self.offset = nn.Parameter(torch.zeros(len(JOINT_NAMES)))   # b_j (14)
        self.velocity_head = nn.Linear(n_dn, 2)                      # NOT MN driven
        with torch.no_grad():
            self.velocity_head.weight.mul_(0.1)
            self.velocity_head.bias.zero_()
        self.body_velocity_source = "DN linear readout (not MN driven)"
        self.fallback_head = None
        if self.fallback_joints:
            self.fallback_head = nn.Linear(n_dn, len(self.fallback_joints))
            self.register_buffer("fallback_index", torch.as_tensor(
                [JOINT_NAMES.index(n) for n in self.fallback_joints]))
        self.out_dim = MN_ACTION_DIM

    def mn_drive(self, runtime, rates: torch.Tensor) -> torch.Tensor:
        """(agonist - antagonist) MN rate sums per joint: ``[B, 14]``."""
        return runtime.rates("MN", rates) @ self.coeff.T

    def forward(self, runtime, rates):
        dn = runtime.rates("DN", rates)
        drive = self.mn_drive(runtime, rates)
        if self.fallback_head is not None:
            drive = drive.clone()
            drive[:, self.fallback_index] = self.fallback_head(dn)
        return torch.cat([self.velocity_head(dn), self.gain * drive + self.offset], dim=1)

    def described(self) -> dict:
        return {"joints": self.joint_names, "mn_counts_agonist_antagonist": self.counts,
                "one_sided": self.one_sided, "fallback_joints": self.fallback_joints,
                "body_velocity_source": self.body_velocity_source,
                "learned_joint_params": int(self.gain.numel() + self.offset.numel())}


register_readout("mn_antagonist", MNAntagonistReadout)


# ============================================================================ per-joint sign verification

# What "positive" means for each DOF in the map ("movement" text) and the geometric proxy used to test it.
# grounding "geometric": the anatomical direction is a plain geometric one (tibia flexion folds the tibia toward the
# femur, tarsus depression moves the claw down, promotion moves the leg forward, abduction moves it outward).
# grounding "assumed": the map gives no geometric direction, the proxy is our reading and needs anatomical review.
PROXIES = {
    "Coxa_yaw": ("geometric", "abduction (+): claw moves away from the midline, d[(tip-coxa).out]"),
    "Coxa": ("geometric", "promotion (+): claw moves forward, d[(tip-coxa).fwd]"),
    "Coxa_roll": ("assumed", "anterior rotation (+): femur-tibia joint moves forward, d[(knee-coxa).fwd]"),
    "Femur": ("assumed", "flexion (+): knee (femur-tibia joint) moves ventrally, -d[knee_z]"),
    "Femur_roll": ("none", "map gives no direction for the reductor MN; sign fixed to +1 by convention"),
    "Tibia": ("geometric", "flexion (+): tibia folds toward the femur, -d|tip - femur_origin|"),
    "Tarsus1": ("geometric", "depression (+): claw moves down, -d[tip_z]"),
}


def _proxy(dof: str, p: dict, fwd: np.ndarray, out: np.ndarray) -> float:
    tip, coxa, femur, knee = p["tip"], p["coxa"], p["femur"], p["tibia"]
    if dof == "Coxa_yaw":
        return float((tip - coxa) @ out)
    if dof in ("Coxa", "Femur_roll"):
        return float((tip - coxa) @ fwd)
    if dof == "Coxa_roll":
        return float((knee - coxa) @ fwd)
    if dof == "Femur":
        return float(-knee[2])
    if dof == "Tibia":
        return -float(np.linalg.norm(tip - femur))
    return float(-tip[2])


def verify_signs(delta: float = 0.2, corr_episodes: int = 6, out_path: Path = SIGN_PATH) -> dict:
    """Move each foreleg joint by ``+delta`` rad from the rest pose in the MuJoCo model (forward kinematics, so claw
    contacts cannot distort it) and measure the anatomical proxy. ``agonist_gain_sign`` is +1 when +q is the map's agonist direction."""
    import mujoco
    from backend.embodiment import STAND_HEIGHT
    from backend.fly_env import FlyTypingEnv
    from backend.physics import LEGS, FlySim

    sim = FlySim()
    m = sim.model
    body_id = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    rest = np.array([sim.fly.rest_q[leg] for leg in LEGS])
    body = np.array([0.0, 0.0, STAND_HEIGHT, 0.0, 0.0, 0.0])   # identity yaw: thorax frame == world frame

    from backend.physics import LegIK
    ik = LegIK(sim)
    d = ik.data

    def pose(q_target: np.ndarray, leg: str) -> dict:
        """Forward kinematics of the MuJoCo model at joint angles ``q_target`` (no contacts involved)."""
        ik.tips(body, q_target)
        return {"tip": d.site_xpos[sim.tip_site[LEGS.index(leg)]].copy(),
                **{k: d.xpos[body_id(f"{leg}{seg}")].copy()
                   for k, seg in (("coxa", "Coxa"), ("femur", "Femur"), ("tibia", "Tibia"))}}

    pose(rest, "RF")   # settle once so body positions are valid
    front = d.xpos[body_id("RFCoxa")] - d.xpos[body_id("RHCoxa")]
    fwd = front.copy()
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    right = d.xpos[body_id("RFCoxa")] - d.xpos[body_id("LFCoxa")]
    right[2] = 0.0
    right /= np.linalg.norm(right)

    result = {}
    for leg in TYPERS:
        out = right if leg == "RF" else -right
        base = pose(rest, leg)
        for di, dof in enumerate(JOINT_DOFS):
            qt = rest.copy()
            qt[LEGS.index(leg), di] += delta
            moved = pose(qt, leg)
            change = (_proxy(dof, moved, fwd, out) - _proxy(dof, base, fwd, out)) / delta
            grounding, text = PROXIES[dof]
            sign = 1 if grounding == "none" or change == 0 else int(np.sign(change))
            result[f"joint_{leg}{dof}"] = {
                "proxy": text, "proxy_grounding": grounding, "proxy_change_per_rad": round(float(change), 4),
                "agonist_gain_sign": sign, "verifiable": grounding != "none"}

    # Cross-check with the scripted expert: correlation of each joint label with the claw action coordinates.
    env = FlyTypingEnv(targets=("fly",), action_mode="mn", terminate_on_error=False)
    rows = {leg: [] for leg in TYPERS}
    for seed, target in enumerate(["Hello", "qpz,M", "fly A", "a1;s", "Zx/.", "wert"][:corr_episodes]):
        env.reset(seed=seed, options={"target": target})
        done = False
        while not done:
            action = env.expert_action()
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc
            for k, leg in enumerate(TYPERS):
                rows[leg].append([*action[2 + 7 * k:9 + 7 * k], *env._claw_coords(leg)])
    for leg in TYPERS:
        arr = np.array(rows[leg])
        for di, dof in enumerate(JOINT_DOFS):
            c = [float(np.corrcoef(arr[:, di], arr[:, 7 + a])[0, 1]) if arr[:, di].std() > 1e-6 else 0.0 for a in range(3)]
            result[f"joint_{leg}{dof}"]["expert_label_corr_with_claw_rx_ry_rz"] = [round(v, 3) for v in c]
    for dof in JOINT_DOFS:
        r, l = result[f"joint_RF{dof}"], result[f"joint_LF{dof}"]
        r["same_sign_as_other_side"] = l["same_sign_as_other_side"] = r["agonist_gain_sign"] == l["agonist_gain_sign"]
    payload = {
        "version": 1,
        "method": f"MuJoCo model forward kinematics at the rest pose with +{delta} rad added to one joint (the position "
                  "actuators track joint targets, so target sign == joint sign); proxy = change in the geometric quantity per rad; sign +1 means +q is the "
                  "map's agonist direction. Thorax frame == world frame (identity yaw).",
        "gain_convention": "g_j initial sign = agonist_gain_sign, so more agonist drive moves the joint in the map's "
                           "nominal positive direction. g_j remains free to change sign in training.",
        "joints": result,
    }
    Path(out_path).write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-signs", action="store_true")
    args = parser.parse_args()
    if args.verify_signs:
        from backend.memory_budget import start_watchdog
        start_watchdog()
        payload = verify_signs()
        for name, r in payload["joints"].items():
            print(f"{name:22s} sign {r['agonist_gain_sign']:+d} d/rad {r['proxy_change_per_rad']:+.3f} "
                  f"[{r['proxy_grounding']}] corr(rx,ry,rz) {r['expert_label_corr_with_claw_rx_ry_rz']}")


if __name__ == "__main__":
    main()
