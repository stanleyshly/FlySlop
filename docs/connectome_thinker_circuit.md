# Thinker connectome circuit (P1b)

Extractor: `scripts/extract_thinker_circuit.py` (reuses `nt_info`, `scan_weights`, `iter_weight_batches` from
`extract_typing_circuit.py` via import; nothing was copied or modified). Output layout matches
`typing_circuit.npz`: CSR with rows = post, `pre/post/weight/edge_sign/indptr`, per-neuron fields, `idx_*`.

    FLYSLOP_MAX_RAM_GB=1.5 uv run --extra connectome python scripts/extract_thinker_circuit.py --preset small|medium

## Annotation names (checked in `body-annotations-male-cns-v1.0-minconf-0.5.feather`, `status == Traced`)

| Population | superclass | Pool size |
|---|---|---|
| VP (input) | `visual_projection` | 9,201 |
| CB (pass-through) | `cb_intrinsic` | 32,160 |
| DN (readout) | `descending_neuron` | 1,314 |

## Selection

1. VP: `default_rng(seed).choice` over the sorted VP body ids (seed 0).
2. CB: score = fwd x bwd, both must be > 0. `fwd` = normalised synapses from the VP sample plus half of the
   2-hop flow (VP -> cb -> this). `bwd` = normalised synapses onto any DN plus half of the 2-hop flow
   (this -> cb -> DN). Top `n_cb` by score, ties by body id.
3. DN: top `n_dn` DNs by synapses received from the selected VP+CB.
4. Edges among the selected set with `weight >= min_weight`, no self loops, then the strongest `topk_in`
   presynaptic partners per postsynaptic neuron (weight desc, pre asc). Edges into VP and out of DN are kept
   (the thinker is recurrent over K micro-steps).
5. Sign: acetylcholine +1, GABA -1, glutamate -1, other named NT +1 (`sign_code` 1), unclear +1 (`sign_code` 2).

## Presets (in `PRESETS`; every value is overridable by CLI flag)

| | small | medium |
|---|---|---|
| n_vp / n_cb / n_dn | 800 / 3,900 / 300 | 2,500 / 13,500 / 1,000 |
| min_weight / topk_in | 2 / 400 | 3 / 100 |
| neurons | 5,000 | 17,000 |
| edges (before topk) | 536,608 (543,934) | 1,048,268 (1,508,516) |
| inhibitory edges | 186,500 | 409,816 |
| npz size | 1.48 MB | 3.38 MB |
| content hash | `2e628ab2bb04...` | `215b9322e641...` |
| DN reachable from VP in K steps | k1 0.63, k2+ 1.00 | k1 0.37, k2+ 1.00 |
| extraction time / peak RSS | ~14 s / 0.94 GB | ~13 s / 1.07 GB |

K = 4 reachability is 1.00 for both DN and CB in both presets. The `.json` next to each npz has the full
per-population counts, edge blocks (VP->CB, CB->CB, CB->DN, ...) and sign-code counts.

## Caveats

- Reachability is structural (any path of length <= K), not weighted signal strength. A DN 2 steps out is
  reachable, but the sample's synaptic drive to it may be small.
- CB->CB dominates the edges (about 80%), as expected for a brain-internal sample. The path score, not the class, picks the CB neurons.
- `topk_in` cap trims about 1% of small-preset edges but 30% of medium-preset edges.
- Presets live in the script, not a shared config file, because no config module exists yet.
