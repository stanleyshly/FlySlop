"""Extract the right-foreleg circuit from MaleCNS v1.0 flat-connectome files.

Inputs (download once, kept out of Git under ``data/full_cns/``; CC-BY,
Janelia FlyEM MaleCNS v1.0)::

    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
        body-annotations-male-cns-v1.0-minconf-0.5.feather
        body-neurotransmitters-male-cns-v1.0.feather
        connectome-weights-male-cns-v1.0-minconf-0.5.feather   (1.05 GB)

Selection (all ``status == Traced`` where annotated):

- MN: ``vnc_motor`` neurons leaving through the prothoracic leg nerve with a
  right soma (right front leg motor neurons), all of them.
- IN: T1 ``vnc_intrinsic`` neurons ranked by synapses onto those MNs plus half
  their input from the SN pool; top ``--interneurons``.
- SN: right prothoracic-leg-nerve proprioceptive and tactile sensory neurons
  ranked by synapses onto the selected IN+MN; top ``--sensory``.
- DN: descending neurons ranked by synapses onto the selected IN+MN; top
  ``--descending``.

Edges among selected neurons with at least ``--min-weight`` synapses are kept.
Sign comes from the consensus predicted neurotransmitter: acetylcholine +1,
GABA and glutamate -1 (glutamate is usually inhibitory in the fly CNS via
GluCl), others +1 (modulatory, flagged). Writes
``data/connectome/rf_leg_circuit.json``.

    uv run --extra connectome python scripts/extract_connectome_circuit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "full_cns"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {"annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
         "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
         "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather"}
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interneurons", type=int, default=192)
    parser.add_argument("--sensory", type=int, default=64)
    parser.add_argument("--descending", type=int, default=64)
    parser.add_argument("--min-weight", type=int, default=3)
    parser.add_argument("--output", default=str(ROOT / "data" / "connectome" / "rf_leg_circuit.json"))
    args = parser.parse_args()

    ann = pd.read_feather(SRC / FILES["annotations"])
    ann = ann[ann["status"].isin(["Traced", "Anchor"]) | ann["superclass"].isin(["vnc_sensory"])]
    mn = ann[(ann.superclass == "vnc_motor") & (ann.exitNerve == "ProLN") & (ann.somaSide == "R")]
    in_pool = ann[(ann.superclass == "vnc_intrinsic") & (ann.somaNeuromere == "T1")]
    sn_pool = ann[(ann.superclass == "vnc_sensory") & (ann.entryNerve == "ProLN") & (ann.rootSide == "R")
                  & ann["class"].isin(["mechanosensory_proprioceptive", "mechanosensory_tactile"])]
    dn_pool = ann[ann.superclass == "descending_neuron"]
    print(f"pools: MN {len(mn)} IN {len(in_pool)} SN {len(sn_pool)} DN {len(dn_pool)}", flush=True)

    ids = {g: set(df.bodyId.astype(np.int64)) for g, df in
           (("MN", mn), ("IN", in_pool), ("SN", sn_pool), ("DN", dn_pool))}
    posts = ids["MN"] | ids["IN"]
    pres = ids["MN"] | ids["IN"] | ids["SN"] | ids["DN"]
    table = feather.read_table(SRC / FILES["weights"], memory_map=True)
    keep = pc.and_(pc.is_in(table["body_post"], value_set=pa.array(sorted(posts), pa.int64())),
                   pc.is_in(table["body_pre"], value_set=pa.array(sorted(pres), pa.int64())))
    w = table.filter(keep).to_pandas()
    print(f"edges into MN/IN pools: {len(w)}", flush=True)

    def into(pre_ids, post_ids):
        sub = w[w.body_pre.isin(pre_ids) & w.body_post.isin(post_ids)]
        return sub.groupby("body_pre").weight.sum()

    premotor = w[w.body_pre.isin(ids["IN"]) & w.body_post.isin(ids["MN"])].groupby("body_pre").weight.sum()
    from_sn = w[w.body_pre.isin(ids["SN"]) & w.body_post.isin(ids["IN"])].groupby("body_post").weight.sum()
    score = premotor.add(0.5 * from_sn.reindex(premotor.index).fillna(0), fill_value=0)
    chosen_in = list(score.sort_values(ascending=False).index[: args.interneurons])
    targets = set(chosen_in) | ids["MN"]
    chosen_sn = list(into(ids["SN"], targets).sort_values(ascending=False).index[: args.sensory])
    chosen_dn = list(into(ids["DN"], targets).sort_values(ascending=False).index[: args.descending])
    chosen_mn = sorted(ids["MN"])

    order = ([("DN", b) for b in chosen_dn] + [("SN", b) for b in chosen_sn]
             + [("IN", b) for b in chosen_in] + [("MN", b) for b in chosen_mn])
    index = {body: i for i, (_g, body) in enumerate(order)}
    nt = pd.read_feather(SRC / FILES["neurotransmitters"], columns=["body", "consensus_nt"]).set_index("body")
    info = ann.set_index("bodyId")
    neurons, signs = [], {}
    for group, body in order:
        row = info.loc[body]
        transmitter = nt["consensus_nt"].get(body) if body in nt.index else None
        transmitter = None if transmitter is None or (isinstance(transmitter, float) and np.isnan(transmitter)) else str(transmitter)
        signs[body] = SIGN.get(transmitter, 1)
        soma = row.get("somaLocation")
        neurons.append({"bodyId": int(body), "group": group,
                        "type": None if pd.isna(row.get("type")) else str(row.get("type")),
                        "instance": None if pd.isna(row.get("instance")) else str(row.get("instance")),
                        "class": None if pd.isna(row.get("class")) else str(row.get("class")),
                        "nt": transmitter, "sign": signs[body],
                        "soma_voxel": None if soma is None or (isinstance(soma, float)) else [int(v) for v in soma]})
    sel = w[w.body_pre.isin(index) & w.body_post.isin(index) & (w.weight >= args.min_weight)]
    edges = [[index[int(r.body_pre)], index[int(r.body_post)], int(r.weight), signs[int(r.body_pre)]]
             for r in sel.itertuples()]
    circuit = {
        "name": "MaleCNS v1.0 right front leg circuit",
        "source": {"dataset": "Janelia FlyEM MaleCNS v1.0", "license": "CC-BY",
                   "attribution": "Male CNS connectome, Janelia Research Campus FlyEM project (male-cns.janelia.org). "
                                  "Cite the MaleCNS v1.0 release when using this subset.",
                   "base_url": BASE, "files": {k: {"name": v, "sha256": sha256(SRC / v)} for k, v in FILES.items()},
                   "extracted": date.today().isoformat()},
        "selection": vars(args) | {"output": Path(args.output).name},
        "sign_rule": "consensus_nt: acetylcholine +1; gaba -1; glutamate -1; other or unknown +1",
        "voxel_nm": 8,
        "neurons": neurons,
        "edges": edges,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(circuit, separators=(",", ":")) + "\n", encoding="utf-8")
    counts = {g: sum(n["group"] == g for n in neurons) for g in ("DN", "SN", "IN", "MN")}
    print(json.dumps({"output": str(out), "neurons": counts, "edges": len(edges),
                      "synapses": int(sum(e[2] for e in edges)),
                      "inhibitory_edges": sum(e[3] < 0 for e in edges)}, indent=2))


if __name__ == "__main__":
    main()
