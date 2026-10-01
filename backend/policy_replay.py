"""Replay a learned policy on the physical fly, with connectome activity if the actor has one."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .connectome import load_circuit

POPULATIONS = ("DN", "SN", "IN", "MN")
TOP_K = 32


def _run_config(checkpoint: Path) -> dict:
    """Training config from the run's manifest (next to the checkpoint), else the fly default."""
    for manifest in (checkpoint.parent / "manifest.json", checkpoint.parent.parent / "manifest.json"):
        if manifest.is_file():
            return json.loads(manifest.read_text())["config"]
    from training.common import load_config
    return load_config(Path(__file__).resolve().parents[1] / "training" / "ppo_fly.json")


def neuron_positions(circuit: dict) -> list[list[float]]:
    """Soma positions (microns, centred); sensory neurons without somata sit near their partners."""
    pos = [np.array(n["soma_voxel"], float) * circuit["voxel_nm"] / 1000 if n["soma_voxel"] else None
           for n in circuit["neurons"]]
    known = np.array([p for p in pos if p is not None])
    centre = known.mean(axis=0)
    partners: dict[int, list[int]] = {}
    for pre, post, _w, _s in circuit["edges"]:
        partners.setdefault(pre, []).append(post)
        partners.setdefault(post, []).append(pre)
    rng = np.random.default_rng(0)
    out = []
    for i, p in enumerate(pos):
        if p is None:
            near = [pos[j] for j in partners.get(i, []) if pos[j] is not None]
            p = (np.mean(near, axis=0) if near else centre) + rng.normal(0, 8, 3)
        out.append((p - centre).round(1).tolist())
    return out


def _actor_of(model):
    """The connectome actor module (new runtime or legacy dense), or None for the MLP policy."""
    actor = getattr(getattr(model.policy, "mlp_extractor", None), "actor", None)
    return actor if actor is not None and hasattr(actor, "last_activity") else None


def _circuit_labels(model, actor) -> tuple[list[str], list[str], list | None]:
    """(group per neuron, type per neuron, soma positions or None) for the actor's circuit."""
    kwargs = getattr(model.policy, "connectome_kwargs", None)
    if kwargs is None:                                   # legacy 360-neuron actor
        circuit = load_circuit()
        return ([n["group"] for n in circuit["neurons"]], [n["type"] for n in circuit["neurons"]],
                neuron_positions(circuit))
    from training.connectome_policy import resolve_circuit
    arrays, _hash = resolve_circuit(kwargs["circuit"])
    n = actor.runtime.n
    groups = ["IN"] * n
    for pop in POPULATIONS:
        for i in arrays.get(f"idx_{pop}", []):
            groups[int(i)] = pop
    if "cell_type" in arrays:
        types = [str(t) for t in arrays["cell_type"]]
    elif str(kwargs["circuit"]).endswith(".json"):
        types = [x["type"] for x in load_circuit(kwargs["circuit"])["neurons"]]
    else:
        types = [""] * n
    if str(kwargs["circuit"]).endswith(".json"):
        positions = neuron_positions(load_circuit(kwargs["circuit"]))
    else:   # no soma coordinates in the typing circuit: deterministic schematic layout by population
        rng = np.random.default_rng(0)
        centre = {"DN": (0, 0, 120), "SN": (0, 0, -120), "IN": (0, 0, 0), "MN": (120, 0, -40)}
        positions = [(np.array(centre[g], float) + rng.normal(0, 45, 3)).round(1).tolist() for g in groups]
    return groups, types, positions


def activity_summary(activity: np.ndarray, groups: list[str], k: int = TOP_K) -> dict:
    """Per-frame population mean |rate| (DN/SN/IN/MN) and the ids of the ``k`` most active neurons.

    ``activity`` is ``[T, n]`` in [-1, 1]. Rates are rounded to 3 decimals to keep replays small."""
    groups_arr = np.array(groups)
    mean = {p: np.abs(activity[:, groups_arr == p]).mean(axis=1).round(3).tolist()
            for p in POPULATIONS if (groups_arr == p).any()}
    k = min(k, activity.shape[1])
    top = np.argsort(-np.abs(activity), axis=1)[:, :k]
    return {"population_mean_abs_rate": mean, "top_k": k, "top_ids": top.astype(int).tolist()}


def policy_episode(checkpoint: Path, target_id: str, text: str, seed: int = 0, max_steps: int | None = None) -> dict:
    """Roll the checkpoint on ``text`` and return the replay. Connectome actors add a ``connectome`` block
    (population mean rates, top-k ids, int8 rates); an MLP policy gives a replay without one."""
    from training.common import make_env
    from training.train_ppo import load_ppo
    config = _run_config(checkpoint)
    model = load_ppo(checkpoint, device="cpu")
    env = make_env(config, [text], record=True)
    obs, _ = env.reset(seed=seed, options={"target": text, "start": (0.0, -0.4)})
    actor = _actor_of(model)
    activity = []
    done, steps = False, 0
    while not done and (max_steps is None or steps < max_steps):
        action, _ = model.predict(obs, deterministic=True)
        if actor is not None and actor.last_activity is not None:
            activity.append(actor.last_activity[0].cpu().numpy())
        obs, _r, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        steps += 1
    replay = env.replay(target_id, "learned_policy_physics_body")
    replay["metadata"]["policy"] = {"checkpoint": checkpoint.name, "environment": config["environment"],
                                    "actor": "connectome" if actor is not None else "mlp"}
    if activity:
        groups, types, positions = _circuit_labels(model, actor)
        kw = getattr(model.policy, "connectome_kwargs", None)
        positions_kind_soma = kw is None or str(kw["circuit"]).endswith(".json")
        activity.append(activity[-1])   # one sample per recorded frame (frame 0 is the reset pose)
        arr = np.array(activity)
        packed = np.clip(np.round(arr * 127), -127, 127).astype(np.int8)
        replay["connectome"] = {
            "label": "MaleCNS v1.0 wiring (CC-BY) · modelled rate activity of the policy's actor",
            "groups": groups,
            "types": types,
            "positions_um": positions,
            "positions_kind": "soma" if positions_kind_soma else "schematic_layout",
            "activity_int8_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
            "activity_shape": list(packed.shape),
            "activity_scale": 1 / 127,
            **activity_summary(arr, groups),
        }
    return replay
