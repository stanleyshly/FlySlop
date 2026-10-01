"""Extract the typing-fly circuit (foreleg MNs + premotor INs + DNs + SNs) from MaleCNS v1.0.

Populations (all ``status == Traced``; see PLAN.md section "Typing connectome"):

- MN: all ProLN foreleg MNs (both sides) plus T1-soma MNs leaving by other nerves (coxa/trochanter
  leg muscles ``subclass fl`` and body-wall muscles), so the coxa joints can be driven.
- IN: top ``--interneurons`` ``vnc_intrinsic`` neurons ranked by synapses to plus from the MN set.
- DN: descending neurons ranked by synapses onto MN+IN, top ``--descending``.
- SN: ProLN proprioceptive and tactile sensory neurons (both sides) ranked by synapses onto MN+IN,
  top ``--sensory``.

Edges among the selected neurons with weight >= ``--min-weight`` are kept. Sign follows the existing
rule (``extract_connectome_circuit.py``): acetylcholine +1, GABA -1, glutamate -1; every other or
``unclear`` transmitter is +1 but flagged in ``sign_code`` (0 rule, 1 other named NT, 2 unclear).

The 151.9M-row weights table is streamed batch by batch from a memory map with column projection;
only edges between candidate ids are retained, so peak RSS stays well under 2 GB.

    FLYSLOP_MAX_RAM_GB=2 uv run --extra connectome python scripts/extract_typing_circuit.py
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import memory_budget as mb  # noqa: E402
from backend.connectome import circuit_io  # noqa: E402

SRC = ROOT / "data" / "full_cns"
OUT_DIR = ROOT / "data" / "connectome"
FILES = {"annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
         "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
         "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather"}
ROLES = ("DN", "SN", "IN", "MN")
RULE_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
SN_CLASSES = ("mechanosensory_proprioceptive", "mechanosensory_tactile")
ANN_COLS = ["bodyId", "superclass", "class", "subclass", "type", "somaSide", "somaNeuromere",
            "entryNerve", "exitNerve", "status"]
DEFAULTS = {"interneurons": 2500, "descending": 300, "sensory": 200, "min_weight": 4}


def load_annotations(path: Path | None = None) -> pd.DataFrame:
    ann = pd.read_feather(path or SRC / FILES["annotations"], columns=ANN_COLS)
    return ann[ann.status == "Traced"].copy()


def select_pools(ann: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Candidate pools before ranking. MN pool is final; IN/DN/SN pools are ranked later."""
    mn = ann[(ann.superclass == "vnc_motor")
             & ((ann.exitNerve == "ProLN") | (ann.somaNeuromere == "T1"))]
    return {
        "MN": mn,
        "IN": ann[ann.superclass == "vnc_intrinsic"],
        "DN": ann[ann.superclass == "descending_neuron"],
        "SN": ann[(ann.superclass == "vnc_sensory") & (ann.entryNerve == "ProLN") & ann["class"].isin(SN_CLASSES)],
    }


def scan_weights(batches, candidates: np.ndarray, min_weight: int):
    """Keep edges with both endpoints in ``candidates`` (sorted int64) and weight >= min_weight."""
    keep_pre, keep_post, keep_w = [], [], []
    for pre, post, w in batches:
        m = w >= min_weight
        pre, post, w = pre[m], post[m], w[m]
        ip = np.searchsorted(candidates, pre)
        ip[ip == len(candidates)] = 0
        io = np.searchsorted(candidates, post)
        io[io == len(candidates)] = 0
        m = (candidates[ip] == pre) & (candidates[io] == post)
        if m.any():
            keep_pre.append(pre[m]); keep_post.append(post[m]); keep_w.append(w[m])
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
    return cat(keep_pre, np.int64), cat(keep_post, np.int64), cat(keep_w, np.int64)


def iter_weight_batches(path: Path):
    import pyarrow as pa
    reader = pa.ipc.open_file(pa.memory_map(str(path)))
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i).select(["body_pre", "body_post", "weight"])
        yield (b.column(0).to_numpy(zero_copy_only=False), b.column(1).to_numpy(zero_copy_only=False),
               b.column(2).to_numpy(zero_copy_only=False))


def _rank(pre, post, w, src_ids, dst_ids, both_ways=False) -> pd.Series:
    """Summed weight from ``src_ids`` onto ``dst_ids`` (optionally plus dst->src) indexed by src id."""
    def agg(a, b):
        m = np.isin(pre, a) & np.isin(post, b)
        return pd.Series(w[m]).groupby(pre[m]).sum()
    s = agg(src_ids, dst_ids)
    if both_ways:
        m = np.isin(pre, dst_ids) & np.isin(post, src_ids)
        s = s.add(pd.Series(w[m]).groupby(post[m]).sum(), fill_value=0)
    return s


def _top(score: pd.Series, k: int) -> list[int]:
    df = pd.DataFrame({"score": score.values, "id": score.index.values.astype(np.int64)})
    df = df[df.score > 0].sort_values(["score", "id"], ascending=[False, True])
    return df.id.head(k).tolist()


def nt_info(nt_table: pd.DataFrame, ids: np.ndarray):
    """(nt string, sign, sign_code) per id. Code 0 = rule (ACh/GABA/Glu), 1 = other named NT, 2 = unclear/missing."""
    lut = nt_table.set_index("body")["consensus_nt"]
    names = lut.reindex(ids).fillna("unclear").astype(str).to_numpy()
    sign = np.array([RULE_SIGN.get(n, 1) for n in names], np.int8)
    code = np.array([0 if n in RULE_SIGN or n == "acetylcholine" else (2 if n == "unclear" else 1) for n in names], np.int8)
    return names, sign, code


def build_circuit(ann: pd.DataFrame, nt_table: pd.DataFrame, edges, cfg: dict) -> tuple[dict, dict]:
    """Pure function: annotations + NT table + scanned edges (pre, post, w) -> (arrays, report)."""
    pools = select_pools(ann)
    pre, post, w = edges
    keep = w >= cfg["min_weight"]
    pre, post, w = pre[keep], post[keep], w[keep]
    mn_ids = np.sort(pools["MN"].bodyId.to_numpy(np.int64))
    in_pool = pools["IN"].bodyId.to_numpy(np.int64)
    in_pool = np.setdiff1d(in_pool, mn_ids)
    chosen_in = np.array(sorted(_top(_rank(pre, post, w, in_pool, mn_ids, both_ways=True), cfg["interneurons"])), np.int64)
    core = np.union1d(mn_ids, chosen_in)
    chosen_dn = np.array(sorted(_top(_rank(pre, post, w, np.setdiff1d(pools["DN"].bodyId.to_numpy(np.int64), core), core), cfg["descending"])), np.int64)
    chosen_sn = np.array(sorted(_top(_rank(pre, post, w, np.setdiff1d(pools["SN"].bodyId.to_numpy(np.int64), core), core), cfg["sensory"])), np.int64)

    order = [(0, chosen_dn), (1, chosen_sn), (2, chosen_in), (3, mn_ids)]   # stable order: role, then bodyId
    body = np.concatenate([ids for _, ids in order]).astype(np.int64)
    role_code = np.concatenate([np.full(len(ids), c, np.int8) for c, ids in order])
    n = len(body)
    info = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(body)
    txt = lambda col: info[col].fillna("").astype(str).to_numpy().astype(str)  # noqa: E731
    is_mn = role_code == 3
    nt_names, sign, sign_code = nt_info(nt_table, body)
    nt_names = nt_names.astype(str)
    mn_group = np.where(~is_mn, "", np.where(txt("exitNerve") == "ProLN", "proln_foreleg",
                        np.where(txt("subclass") == "fl", "t1_leg_other_nerve", "t1_bodywall"))).astype(str)

    # edges among selected, local indices, sorted by (post, pre) -> CSR over rows=post
    sorter = np.argsort(body)
    def local(ids):
        pos = np.searchsorted(body[sorter], ids); pos[pos == n] = 0
        ok = body[sorter][pos] == ids
        return sorter[pos], ok
    lp, okp = local(pre); lo, oko = local(post)
    m = okp & oko & (pre != post)
    lp, lo, lw = lp[m].astype(np.int32), lo[m].astype(np.int32), w[m].astype(np.int32)
    o = np.lexsort((lp, lo))
    lp, lo, lw = lp[o], lo[o], lw[o]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, lo + 1, 1)
    indptr = np.cumsum(indptr)

    mn_side = np.where(is_mn, txt("somaSide"), "")
    arrays = {
        "body_id": body, "role_code": role_code, "role": np.array(ROLES)[role_code],
        "cell_type": txt("type"), "soma_side": txt("somaSide"), "soma_neuromere": txt("somaNeuromere"),
        "mn_type": np.where(is_mn, txt("type"), ""), "mn_side": mn_side,
        "mn_nerve": np.where(is_mn, txt("exitNerve"), ""), "mn_group": mn_group,
        "nt": nt_names, "sign": sign, "sign_code": sign_code,
        "pre": lp, "post": lo, "weight": lw, "edge_sign": sign[lp], "indptr": indptr,
    }
    for c, r in enumerate(ROLES):
        arrays[f"idx_{r}"] = np.flatnonzero(role_code == c).astype(np.int32)
    arrays["idx_MN_L"] = np.flatnonzero(is_mn & (mn_side == "L")).astype(np.int32)
    arrays["idx_MN_R"] = np.flatnonzero(is_mn & (mn_side == "R")).astype(np.int32)
    arrays["idx_MN_ProLN"] = np.flatnonzero(mn_group == "proln_foreleg").astype(np.int32)

    edge_role = role_code[lp] * 4 + role_code[lo]
    per_pop = {r: {"neurons": int((role_code == c).sum()),
                   "edges_in": int((role_code[lo] == c).sum()), "edges_out": int((role_code[lp] == c).sum()),
                   "signs_neg": int((sign[role_code == c] < 0).sum()),
                   "sign_code_counts": {str(k): int(((sign_code == k) & (role_code == c)).sum()) for k in (0, 1, 2)}}
              for c, r in enumerate(ROLES)}
    block = {f"{ROLES[a]}->{ROLES[b]}": int((edge_role == a * 4 + b).sum()) for a in range(4) for b in range(4) if (edge_role == a * 4 + b).any()}
    report = {"neurons": n, "edges": int(len(lp)), "synapses": int(lw.sum()),
              "inhibitory_edges": int((arrays["edge_sign"] < 0).sum()),
              "populations": per_pop, "edge_blocks": block,
              "mn_groups": {g: int((mn_group == g).sum()) for g in sorted(set(mn_group[is_mn]))},
              "mn_types": {(t or "(untyped)"): int((arrays["mn_type"][is_mn] == t).sum()) for t in sorted(set(arrays["mn_type"][is_mn].tolist()))}}
    return arrays, report


def joint_coverage(arrays: dict, joint_map: dict) -> dict:
    """Per joint: agonist/antagonist MN counts (side-matched) and their in-circuit synaptic input."""
    mn = arrays["role"] == "MN"
    indeg = np.bincount(arrays["post"], weights=arrays["weight"], minlength=len(arrays["body_id"]))
    out = {}
    for name, j in joint_map["joints"].items():
        side_ok = mn & (arrays["mn_side"] == j["side"])
        row = {"status": j["status"], "confidence": j.get("confidence", "")}
        for role in ("agonist", "antagonist"):
            sel = side_ok & np.isin(arrays["mn_type"], j[role])
            row[role] = {"types": list(j[role]), "n_mn": int(sel.sum()), "input_synapses": int(indeg[sel].sum())}
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interneurons", type=int, default=DEFAULTS["interneurons"])
    ap.add_argument("--descending", type=int, default=DEFAULTS["descending"])
    ap.add_argument("--sensory", type=int, default=DEFAULTS["sensory"])
    ap.add_argument("--min-weight", type=int, default=DEFAULTS["min_weight"])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--edge-cache", default=None, help="optional npz of scanned candidate edges (skips the 152M-row scan)")
    ap.add_argument("--output", default=str(OUT_DIR / "typing_circuit.npz"))
    args = ap.parse_args()
    mb.start_watchdog()
    t0 = time.time()
    src = Path(args.src)
    cfg = {"interneurons": args.interneurons, "descending": args.descending, "sensory": args.sensory,
           "min_weight": args.min_weight}
    ann = load_annotations(src / FILES["annotations"])
    nt = pd.read_feather(src / FILES["neurotransmitters"], columns=["body", "consensus_nt"])
    pools = select_pools(ann)
    print("pools:", {k: len(v) for k, v in pools.items()}, flush=True)
    cand = np.unique(np.concatenate([p.bodyId.to_numpy(np.int64) for p in pools.values()]))
    cache = Path(args.edge_cache) if args.edge_cache else None
    if cache and cache.exists():
        z = np.load(cache); edges = (z["pre"], z["post"], z["w"])
    else:
        edges = scan_weights(iter_weight_batches(src / FILES["weights"]), cand, cfg["min_weight"])
        if cache:
            np.savez(cache, pre=edges[0], post=edges[1], w=edges[2])
    t_scan = time.time() - t0
    print(f"candidate edges: {len(edges[0])} ({t_scan:.0f} s, rss {mb.peak_rss() / 2**30:.2f} GB)", flush=True)
    arrays, report = build_circuit(ann, nt, edges, cfg)
    meta = {"name": "MaleCNS v1.0 typing circuit (bilateral foreleg)", "config": cfg,
            "sign_rule": "consensus_nt: acetylcholine +1; gaba -1; glutamate -1; other named NT +1 (sign_code 1); unclear +1 (sign_code 2)",
            "source": {"dataset": "Janelia FlyEM MaleCNS v1.0", "license": "CC-BY",
                       "files": FILES, "attribution": "Male CNS connectome, Janelia Research Campus FlyEM project."},
            "csr": "rows = post, cols = pre, sorted by (post, pre); indptr length n+1"}
    digest = circuit_io.save_circuit(args.output, arrays, meta)
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, mb.peak_rss())  # bytes on macOS
    summary = {"npz": Path(args.output).name, "content_hash": digest, "config": cfg, "extracted": date.today().isoformat(),
               "extraction_seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak / 2**30, 3),
               "ram_cap_gb": mb.max_ram_bytes() / 2**30, **report}
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("mn_types",)}, indent=1))


if __name__ == "__main__":
    main()
