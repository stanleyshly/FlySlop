"""Extract the thinker circuit (visual_projection sample -> cb_intrinsic -> descending neurons) from MaleCNS v1.0.

Populations (all ``status == Traced``; see PLAN.md section "Thinker connectome"):

- VP: seeded sample of ``visual_projection`` neurons (token-embedding inputs).
- CB: ``cb_intrinsic`` neurons ranked by a two-hop path score: (synapses received from the VP sample,
  directly plus via one cb hop) x (synapses sent to any DN, directly plus via one cb hop). Both terms
  must be > 0.
- DN: ``descending_neuron`` readout, top-N by synapses received from the selected VP+CB.

Edges among selected neurons with weight >= ``min_weight`` are kept, then capped to the strongest
``topk_in`` presynaptic partners per postsynaptic neuron (deterministic tie-break on pre id). Sign rule,
``sign_code`` and the streamed weight scan are those of ``extract_typing_circuit.py`` (imported, not
modified). Output layout matches ``typing_circuit.npz`` (CSR rows = post).

    FLYSLOP_MAX_RAM_GB=1.5 uv run --extra connectome python scripts/extract_thinker_circuit.py --preset small
"""

from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location("extract_typing_circuit", ROOT / "scripts" / "extract_typing_circuit.py")
_p1a = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("extract_typing_circuit", _p1a)
_spec.loader.exec_module(_p1a)
nt_info, scan_weights, iter_weight_batches = _p1a.nt_info, _p1a.scan_weights, _p1a.iter_weight_batches
SRC, FILES, OUT_DIR = _p1a.SRC, _p1a.FILES, _p1a.OUT_DIR

ROLES = ("VP", "CB", "DN")
SUPERCLASS = {"VP": "visual_projection", "CB": "cb_intrinsic", "DN": "descending_neuron"}
ANN_COLS = ["bodyId", "superclass", "class", "type", "somaSide", "somaNeuromere", "status"]
K_STEPS = 4
# Size presets. Totals: small 5000, medium 17000 neurons.
PRESETS = {
    "small": {"n_vp": 800, "n_cb": 3900, "n_dn": 300, "min_weight": 2, "topk_in": 400, "seed": 0},
    "medium": {"n_vp": 2500, "n_cb": 13500, "n_dn": 1000, "min_weight": 3, "topk_in": 100, "seed": 0},
}
SCAN_MIN_WEIGHT = 2     # scan floor; the scan uses min(this, preset min_weight)


def load_annotations(path: Path | None = None) -> pd.DataFrame:
    ann = pd.read_feather(path or SRC / FILES["annotations"], columns=ANN_COLS)
    return ann[ann.status == "Traced"].copy()


def select_pools(ann: pd.DataFrame) -> dict[str, np.ndarray]:
    return {r: np.sort(ann.loc[ann.superclass == sc, "bodyId"].to_numpy(np.int64)) for r, sc in SUPERCLASS.items()}


def _lookup(sorted_ids: np.ndarray, ids: np.ndarray):
    """Position of each of ``ids`` in ``sorted_ids`` and a membership mask."""
    if len(sorted_ids) == 0:
        return np.zeros(len(ids), np.int64), np.zeros(len(ids), bool)
    pos = np.searchsorted(sorted_ids, ids)
    pos[pos == len(sorted_ids)] = 0
    return pos, sorted_ids[pos] == ids


def path_scores(cb_ids, vp_ids, dn_ids, pre, post, w):
    """(fwd, bwd) per cb neuron: 1-hop plus 2-hop (via cb) synapse flow from VP sample / to any DN."""
    ncb = len(cb_ids)
    ip, okp = _lookup(cb_ids, pre)
    io, oko = _lookup(cb_ids, post)
    from_vp = np.isin(pre, vp_ids) & oko
    to_dn = okp & np.isin(post, dn_ids)
    cc = okp & oko & (pre != post)
    wf = w.astype(np.float64)
    f1 = np.bincount(io[from_vp], weights=wf[from_vp], minlength=ncb)
    b1 = np.bincount(ip[to_dn], weights=wf[to_dn], minlength=ncb)
    f1n, b1n = f1 / max(f1.max(), 1), b1 / max(b1.max(), 1)
    f2 = np.bincount(io[cc], weights=wf[cc] * f1n[ip[cc]], minlength=ncb)     # vp -> cb -> this
    b2 = np.bincount(ip[cc], weights=wf[cc] * b1n[io[cc]], minlength=ncb)     # this -> cb -> dn
    fwd, bwd = f1n + f2 / max(f2.max(), 1e-12) * 0.5, b1n + b2 / max(b2.max(), 1e-12) * 0.5
    return fwd, bwd


def _top(ids: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    order = np.lexsort((ids, -score))      # score desc, id asc
    order = order[score[order] > 0][:k]
    return np.sort(ids[order])


def reachability(arrays: dict, k_max: int = 6) -> dict:
    """Fraction of DN (and CB) neurons reachable from the VP inputs in exactly-at-most k steps (k=1..k_max)."""
    n = len(arrays["body_id"])
    pre, post = arrays["pre"], arrays["post"]
    front = np.zeros(n, bool); front[arrays["idx_VP"]] = True
    seen = front.copy()
    out = {}
    for k in range(1, k_max + 1):
        nxt = np.zeros(n, bool); nxt[post[front[pre]]] = True
        front = nxt & ~seen
        seen |= nxt
        out[str(k)] = {"DN": float(seen[arrays["idx_DN"]].mean()) if len(arrays["idx_DN"]) else 0.0,
                       "CB": float(seen[arrays["idx_CB"]].mean()) if len(arrays["idx_CB"]) else 0.0}
    return out


def build_circuit(ann: pd.DataFrame, nt_table: pd.DataFrame, edges, cfg: dict) -> tuple[dict, dict]:
    """Pure function: annotations + NT table + scanned edges (pre, post, w) -> (arrays, report)."""
    pools = select_pools(ann)
    pre, post, w = edges
    keep = w >= cfg["min_weight"]
    pre, post, w = pre[keep], post[keep], w[keep]
    rng = np.random.default_rng(cfg["seed"])
    vp = np.sort(rng.choice(pools["VP"], size=min(cfg["n_vp"], len(pools["VP"])), replace=False))
    cb_pool = np.setdiff1d(pools["CB"], vp)
    fwd, bwd = path_scores(cb_pool, vp, pools["DN"], pre, post, w)
    cb = _top(cb_pool, np.where((fwd > 0) & (bwd > 0), fwd * bwd, 0.0), cfg["n_cb"])
    core = np.union1d(vp, cb)
    dn_pool = np.setdiff1d(pools["DN"], core)
    ip, okp = _lookup(core, pre)
    idn, okd = _lookup(dn_pool, post)
    m = okp & okd
    dn_in = np.bincount(idn[m], weights=w[m].astype(np.float64), minlength=len(dn_pool))
    dn = _top(dn_pool, dn_in, cfg["n_dn"])

    order = [(0, vp), (1, cb), (2, dn)]
    body = np.concatenate([ids for _, ids in order]).astype(np.int64)
    role_code = np.concatenate([np.full(len(ids), c, np.int8) for c, ids in order])
    n = len(body)
    sorter = np.argsort(body)
    sb = body[sorter]

    def local(ids):
        pos, ok = _lookup(sb, ids)
        return sorter[pos], ok
    lp, okp = local(pre); lo, oko = local(post)
    m = okp & oko & (pre != post)
    lp, lo, lw = lp[m].astype(np.int32), lo[m].astype(np.int32), w[m].astype(np.int32)
    n_before_cap = len(lp)
    # cap: strongest topk_in presynaptic partners per post (weight desc, pre asc)
    o = np.lexsort((lp, -lw, lo))
    lp, lo, lw = lp[o], lo[o], lw[o]
    start = np.searchsorted(lo, np.arange(n))
    rank = np.arange(len(lo)) - start[lo]
    m = rank < cfg["topk_in"]
    lp, lo, lw = lp[m], lo[m], lw[m]
    o = np.lexsort((lp, lo))                        # CSR order: (post, pre)
    lp, lo, lw = lp[o], lo[o], lw[o]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(lo, minlength=n))]).astype(np.int64)

    info = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(body)
    txt = lambda col: info[col].fillna("").astype(str).to_numpy().astype(str)  # noqa: E731
    nt_names, sign, sign_code = nt_info(nt_table, body)
    arrays = {
        "body_id": body, "role_code": role_code, "role": np.array(ROLES)[role_code],
        "cell_type": txt("type"), "cls": txt("class"), "soma_side": txt("somaSide"), "soma_neuromere": txt("somaNeuromere"),
        "nt": nt_names.astype(str), "sign": sign, "sign_code": sign_code,
        "pre": lp, "post": lo, "weight": lw, "edge_sign": sign[lp], "indptr": indptr,
    }
    for c, r in enumerate(ROLES):
        arrays[f"idx_{r}"] = np.flatnonzero(role_code == c).astype(np.int32)

    edge_role = role_code[lp].astype(int) * 3 + role_code[lo]
    per_pop = {r: {"neurons": int((role_code == c).sum()),
                   "edges_in": int((role_code[lo] == c).sum()), "edges_out": int((role_code[lp] == c).sum()),
                   "signs_neg": int((sign[role_code == c] < 0).sum()),
                   "sign_code_counts": {str(k): int(((sign_code == k) & (role_code == c)).sum()) for k in (0, 1, 2)}}
              for c, r in enumerate(ROLES)}
    block = {f"{ROLES[a]}->{ROLES[b]}": int((edge_role == a * 3 + b).sum()) for a in range(3) for b in range(3) if (edge_role == a * 3 + b).any()}
    reach = reachability(arrays)
    report = {"neurons": n, "edges": int(len(lp)), "edges_before_topk": int(n_before_cap), "synapses": int(lw.sum()),
              "inhibitory_edges": int((arrays["edge_sign"] < 0).sum()),
              "populations": per_pop, "edge_blocks": block,
              "reachable_from_inputs": reach, "dn_reachable_k4": reach[str(K_STEPS)]["DN"],
              "dn_reachable_frac_by_k": {k: v["DN"] for k, v in reach.items()},
              "max_indegree": int(np.diff(indptr).max())}
    return arrays, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    for k in ("n_vp", "n_cb", "n_dn", "min_weight", "topk_in", "seed"):
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--edge-cache", default=None, help="optional npz of scanned candidate edges (skips the 152M-row scan)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    mb.start_watchdog()
    t0 = time.time()
    cfg = dict(PRESETS[args.preset])
    cfg.update({k: getattr(args, k) for k in cfg if getattr(args, k) is not None})
    output = Path(args.output or OUT_DIR / f"thinker_circuit_{args.preset}.npz")
    src = Path(args.src)
    ann = load_annotations(src / FILES["annotations"])
    nt = pd.read_feather(src / FILES["neurotransmitters"], columns=["body", "consensus_nt"])
    pools = select_pools(ann)
    print("pools:", {k: len(v) for k, v in pools.items()}, flush=True)
    cand = np.unique(np.concatenate(list(pools.values())))
    scan_min = min(SCAN_MIN_WEIGHT, cfg["min_weight"])
    cache = Path(args.edge_cache) if args.edge_cache else None
    if cache and cache.exists():
        z = np.load(cache); edges = (z["pre"], z["post"], z["w"])
    else:
        edges = scan_weights(iter_weight_batches(src / FILES["weights"]), cand, scan_min)
        if cache:
            np.savez(cache, pre=edges[0], post=edges[1], w=edges[2])
    print(f"candidate edges: {len(edges[0])} ({time.time() - t0:.0f} s, rss {mb.peak_rss() / 2**30:.2f} GB)", flush=True)
    arrays, report = build_circuit(ann, nt, edges, cfg)
    meta = {"name": f"MaleCNS v1.0 thinker circuit ({args.preset})", "config": cfg, "k_steps": K_STEPS,
            "sign_rule": "consensus_nt: acetylcholine +1; gaba -1; glutamate -1; other named NT +1 (sign_code 1); unclear +1 (sign_code 2)",
            "source": {"dataset": "Janelia FlyEM MaleCNS v1.0", "license": "CC-BY",
                       "files": FILES, "attribution": "Male CNS connectome, Janelia Research Campus FlyEM project."},
            "csr": "rows = post, cols = pre, sorted by (post, pre); indptr length n+1"}
    digest = circuit_io.save_circuit(output, arrays, meta)
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, mb.peak_rss())  # bytes on macOS
    summary = {"npz": output.name, "npz_mb": round(output.stat().st_size / 2**20, 2), "content_hash": digest, "preset": args.preset,
               "config": cfg, "extracted": date.today().isoformat(), "extraction_seconds": round(time.time() - t0, 1),
               "peak_rss_gb": round(peak / 2**30, 3), "ram_cap_gb": mb.max_ram_bytes() / 2**30, **report}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
