"""Export the NeuroMechFly v2 body (FlyGym, Apache 2.0) for FlySlop.

Writes ``data/neuromechfly/model.json`` (kinematic tree, joint axes, rest
posture, mesh index) and ``data/neuromechfly/meshes.bin`` (decimated,
little-endian float32 vertices + uint32 faces). The backend uses the tree for
forward/inverse kinematics; the viewer renders the same tree and meshes.

Only needed when regenerating the committed assets:

    uv run --with flygym==1.2.1 --with fast-simplification --with networkx python scripts/export_neuromechfly.py

The exported forward kinematics are checked against MuJoCo before writing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np
import yaml
from flygym import get_data_path
from flygym.examples.locomotion.steps import PreprogrammedSteps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "neuromechfly"
sys.path.insert(0, str(ROOT))

MJCF = "neuromechfly_seqik_kinorder_ypr.xml"
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
LEG_DOFS = ["Coxa_yaw", "Coxa", "Coxa_roll", "Femur", "Femur_roll", "Tibia", "Tarsus1"]
FACE_BUDGET = 90_000


def name(model, kind, index):
    return mujoco.mj_id2name(model, kind, index)


def compile_model(data_dir: Path) -> mujoco.MjModel:
    # Keep jointless bodies (Thorax, abdomen, wings) as separate frames.
    spec = mujoco.MjSpec.from_file(str(data_dir / "mjcf" / MJCF))
    spec.compiler.fusestatic = False
    return spec.compile()


def main() -> None:


    data_dir = get_data_path("flygym", "data")
    model = compile_model(data_dir)
    rest = {k: float(np.deg2rad(v)) for k, v in
            yaml.safe_load((data_dir / "pose" / "pose_tripod.yaml").read_text())["joints"].items()}
    steps = PreprogrammedSteps()
    for leg in LEGS:
        for dof, value in zip(steps.dofs_per_leg, steps.neutral_pos[leg][:, 0]):
            rest[f"joint_{leg}{dof}"] = float(value)

    total_faces = int(model.nmeshface)
    ratio = min(1.0, FACE_BUDGET / total_faces)
    blob = bytearray()
    meshes = []
    for m in range(model.nmesh):
        va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
        fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
        verts = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float32)
        faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int64)
        target = max(160, int(fn * ratio))
        if fn > target:
            verts, faces = fast_simplification.simplify(verts, faces.astype(np.int32), 1 - target / fn)
        verts = np.ascontiguousarray(verts, dtype="<f4")
        faces = np.ascontiguousarray(faces, dtype="<u4")
        entry = {"name": name(model, mujoco.mjtObj.mjOBJ_MESH, m), "vertex_offset": len(blob), "vertex_count": len(verts)}
        blob += verts.tobytes()
        entry.update(face_offset=len(blob), face_count=len(faces))
        blob += faces.tobytes()
        meshes.append(entry)

    bodies = []
    for b in range(1, model.nbody):
        joints = []
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            jn = name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            joints.append({"name": jn, "axis": model.jnt_axis[j].round(6).tolist(),
                           "pos": model.jnt_pos[j].round(6).tolist(), "rest": round(rest.get(jn, 0.0), 6)})
        geoms = [{"mesh": int(model.geom_dataid[g]), "pos": model.geom_pos[g].round(6).tolist(),
                  "quat": model.geom_quat[g].round(6).tolist()}
                 for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b])
                 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        parent = int(model.body_parentid[b])
        bodies.append({"name": name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                       "parent": name(model, mujoco.mjtObj.mjOBJ_BODY, parent) if parent else None,
                       "pos": model.body_pos[b].round(6).tolist(), "quat": model.body_quat[b].round(6).tolist(),
                       "joints": joints, "geoms": geoms})

    # Claw tip: the Tarsus5 mesh vertex farthest from its body origin.
    tips = {}
    by_name = {body["name"]: body for body in bodies}
    for leg in LEGS:
        geom = by_name[f"{leg}Tarsus5"]["geoms"][0]
        mid = geom["mesh"]
        verts = np.asarray(model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(geom["quat"], dtype=float))
        local = verts @ rot.reshape(3, 3).T + np.asarray(geom["pos"])
        tips[leg] = local[np.argmax(np.linalg.norm(local, axis=1))].round(6).tolist()

    payload = {
        "source": {"package": "flygym", "version": importlib.metadata.version("flygym"), "mjcf": MJCF,
                   "license": "Apache-2.0", "units": "mm",
                   "frame": "x forward (head), y left, z up",
                   "rest_pose": "pose_tripod.yaml + PreprogrammedSteps neutral leg DoFs (phase pi)",
                   "mesh_faces_original": total_faces, "mesh_faces_exported": sum(m["face_count"] for m in meshes)},
        "legs": LEGS, "leg_dofs": LEG_DOFS, "tips": tips, "bodies": bodies, "meshes": meshes,
        "meshes_sha256": hashlib.sha256(bytes(blob)).hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.json").write_text(json.dumps(payload, separators=(",", ":")))
    (OUT / "meshes.bin").write_bytes(bytes(blob))
    check_against_mujoco(model)
    print(f"wrote {OUT} ({len(blob) / 1e6:.2f} MB meshes, {payload['source']['mesh_faces_exported']} faces)")


def check_against_mujoco(model) -> None:
    from backend.flybody import FlyBody

    fly = FlyBody()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for _ in range(5):
        angles = {}
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                angles[name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(rng.uniform(-0.8, 0.8))
                data.qpos[model.jnt_qposadr[j]] = angles[name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        mujoco.mj_kinematics(model, data)
        # Thorax is the tree root in FlySlop; compare in its frame.
        thorax = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
        root_pos, root_rot = data.xpos[thorax], data.xmat[thorax].reshape(3, 3)
        for leg in LEGS:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")
            expected = root_rot.T @ (data.xpos[body] - root_pos)
            got = fly.leg_chain_positions(leg, angles)[-1]
            if not np.allclose(expected, got, atol=1e-5):
                raise SystemExit(f"FK mismatch for {leg}: {expected} vs {got}")
    print("forward kinematics match MuJoCo")


if __name__ == "__main__":
    main()
