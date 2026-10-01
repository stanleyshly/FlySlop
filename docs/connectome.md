# Connectome

Status: the curriculum code is implemented; **nothing is fully trained and no
connectome benefit is claimed** (see [results.md](results.md#known-limitations)).
The wiring is measured (Janelia MaleCNS v1.0, CC-BY); everything the networks
do with it is modelled and simulated.

## Circuits used by the curriculum

| Circuit | File | Size | Used by |
| --- | --- | --- | --- |
| Typing circuit | `data/connectome/typing_circuit.npz` | 3,173 neurons, 98.5k edges | actor on the fly body (DN/SN/IN/MN) |
| Thinker, small | `data/connectome/thinker_circuit_small.npz` | 5k neurons, 537k edges | code-writing "thinker" |
| Thinker, medium | `data/connectome/thinker_circuit_medium.npz` | 17k neurons, 1.05M edges | same, larger |

Extraction notes: [connectome_typing_circuit.md](connectome_typing_circuit.md),
[connectome_thinker_circuit.md](connectome_thinker_circuit.md). Circuits load
through `backend/connectome/circuit_io.load_circuit`, which verifies the hash.

**Runtime.** `backend/connectome/runtime.py` `SparseRecurrent` runs the wiring as
a sparse recurrent net (backends csr, index_add, dense; cached CSR, threaded,
sampled_addmm backward). Variants: `real`, `shuffled`, `random_sparse`,
`frozen`. Benchmark: `scripts/bench_runtime.py`.

**Thinker.** `backend/connectome/thinker.py` `build_model(kind)` with kinds
real, shuffled, random_sparse, frozen, and a `gru` baseline, matched at about
681k trainable parameters. The thinker is about 100x slower than the GRU.

**Motor neurons.** With `FlyTypingEnv(action_mode="mn")` the action is 16-D
(vx, vy, 14 joints). `training/mn_readout.py` (`mn_antagonist`, 28 parameters)
maps MN activity to joints with no free linear readout, via
`data/connectome/mn_joint_map.json`. Three joint signs are unverified
(`mn_joint_sign_check.json`).

**Viewer.** A connectome replay carries `population_mean_abs_rate`, `top_ids`,
`positions_kind`, and `activity_shape`. The typing circuit has no soma
coordinates, so `positions_kind` is `schematic_layout` and the panel says so.
The panel labels wiring as measured and activity as model, simulated.

## Legacy: right-foreleg circuit and ablation (package E)

The sections below describe the earlier 360-neuron right-foreleg actor
(`data/connectome/rf_leg_circuit.json`, `training/connectome_policy.py`) and
its ablation protocol. They remain valid for `--policy connectome` but predate
the curriculum.

## Data

The source is Janelia FlyEM **MaleCNS v1.0**, licensed CC-BY
([male-cns.janelia.org](https://male-cns.janelia.org/download/)). Three flat
files are downloaded once into `data/full_cns/` (git-ignored, about 1.1 GB):
body annotations, body neurotransmitters, and connectome weights (minconf 0.5).
`scripts/extract_connectome_circuit.py` writes the derived subset
`data/connectome/rf_leg_circuit.json` (134 KB, committed). It records source
file names, their SHA-256, the selection parameters, and attribution.

```sh
uv run --extra connectome python scripts/extract_connectome_circuit.py
```

## Circuit (right front leg)

Both forelegs type in `FlyTypingEnv`. The actor still uses this right-foreleg
circuit, and its linear readout drives both legs' action blocks. A mirrored
left-foreleg circuit has not been extracted, so LF control is not
connectome-specific.

| Group | Selection | Count |
| --- | --- | --- |
| MN | `vnc_motor`, exit nerve ProLN, right soma: all right-foreleg motor neurons | 40 |
| IN | T1 `vnc_intrinsic`, ranked by synapses onto those MNs (+½ × input from SN pool) | 192 |
| SN | right ProLN proprioceptive + tactile sensory neurons, ranked by output onto IN+MN | 64 |
| DN | descending neurons ranked by output onto IN+MN | 64 |

The subset keeps edges with ≥ 3 synapses: 5,549 edges and 92,900 synapses.
Signs come from the consensus predicted transmitter: acetylcholine +1, GABA
−1, glutamate −1, and other or unknown +1. 2,395 edges are inhibitory. The
glutamate-inhibitory rule and the +1 default are assumptions.

## Model (`training/connectome_policy.py`)

The actor is a rate network over the 360 neurons:
`r ← tanh(W r + drive + b)`, unrolled 3 steps.

- **Weights:** W keeps the variant's sparsity and sign. Only magnitudes are
  learned, initialized from synapse counts.
- **Inputs:** descending neurons get a learned projection of task
  observations (key offset, progress, …). Sensory neurons get proprioceptive
  and tactile observations (claw height and velocity, key depression,
  holding, body velocity). This observation-to-cell assignment is a
  modelling choice.
- **Output:** actions are a linear readout from the DN and MN populations.
- **Critic:** an ordinary MLP in every variant.

What is measured is the **wiring**. Activity, magnitudes, and readouts are
**modelled**, and the viewer labels them that way.

## Ablation protocol (`training/connectome_experiment.py`)

All variants share the demonstrations, the imitation budget, the seeds, and
the held-out evaluation episodes:

- `mlp`: reference 256×256 actor
- `connectome`: measured wiring
- `shuffled`: per-block permutation of measured weights (same counts, signs, and weights)
- `random_sparse`: same density, random positions
- `dense`: all allowed blocks, no sign prior
- `silenced`: connectome with proprioception zeroed
- `random`: uniform random actions

The predeclared metric is held-out exact match. The default is a low-data
regime (60 demo episodes) to avoid a ceiling. The report has per-seed rows,
median and range, and paired bootstrap 95% intervals for connectome minus
each control. A connectome benefit is claimed **only** if the intervals
against `shuffled`, `random_sparse`, and `dense` all exclude zero.

```sh
uv run --extra training --extra connectome python -m training.connectome_experiment \
    --seeds 0 1 2 3 4 --out runs/connectome/exp1
```

## Status

The pipeline was checked with a 2-seed smoke run at a tiny budget (30 demos,
10 epochs). No variant learned at that budget, so no comparison is possible
yet. On a separate imitation run with 240 demos, the connectome actor reached
0.80 held-out exact on 20 episodes, versus 1.00 for the MLP. In the smoke
run, connectome and shuffled wiring reached nearly identical validation loss.
This suggests that, in this setup, the learned encoders and readout may
dominate the wiring. **No connectome benefit is claimed.** Run the full
protocol and report the result whatever it is.
