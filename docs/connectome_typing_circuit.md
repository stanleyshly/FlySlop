# Typing circuit extraction (P1a): review note and coverage report

Artifacts (regenerate with `FLYSLOP_MAX_RAM_GB=2 uv run --extra connectome python scripts/extract_typing_circuit.py`):

- `data/connectome/typing_circuit.npz` (loaded with `backend/connectome/circuit_io.load_circuit`, hash verified on load)
- `data/connectome/typing_circuit.json` (summary: counts, hash, config, time, peak RSS)
- `data/connectome/mn_joint_map.json` (MN type to joint map, reviewed below)

Source: Janelia FlyEM MaleCNS v1.0 flat connectome, `minconf-0.5`, `status == Traced` only (CC-BY).

## Recorded numbers

- **Content hash (sha256 of all arrays):** `49ba419838b367fd4c641c03102e4febea2a5ebcb2df706acccddefe0bfa3d70`. Re-running gives the identical hash (checked several times, also from a different output path). Metadata, timings and file paths are not hashed.
- **Config:** interneurons 2500, descending 300, sensory 200, min_weight 4 (edges with fewer than 4 synapses dropped; 3 gives 124.8k edges, 2 gives 173.8k).
- **Totals:** 3173 neurons, 98,551 edges, 1,710,532 synapses, 45,632 inhibitory edges (46%).
- **Cost:** 4.0 s wall, peak RSS 1.07 GB by `memory_budget.peak_rss` (`/usr/bin/time -l`: 1.18 GB; this includes file-backed pages of the memory-mapped 152M-row weights file, which are reclaimable). Cap used: `FLYSLOP_MAX_RAM_GB=2`. Only 1.34M candidate edges (both endpoints in the 14.9k candidate ids, weight >= 2) are ever held in RAM.

| population | neurons | edges in | edges out | negative-sign neurons |
|---|---|---|---|---|
| DN | 300 | 5,459 | 19,806 | 79 |
| SN | 200 | 186 | 1,671 | 0 |
| IN | 2500 | 76,621 | 76,997 | 1,229 |
| MN | 173 | 16,285 | 77 | 111 |

Edge blocks: IN->IN 61,389; IN->MN 14,446; DN->IN 13,751; DN->DN 4,399; DN->MN 1,656; SN->IN 1,449; SN->MN 139; IN->DN 1,054; others under 110.

MN set (173): 81 ProLN foreleg MNs (`proln_foreleg`, 41 L / 40 R), 54 T1-soma leg MNs leaving through ProAN/VProN (`t1_leg_other_nerve`: coxa/trochanter/sternal muscles), and 38 T1 body-wall MNs (DProN, PDMNa, CvN; `t1_bodywall`, wing/neck/dorsal muscles). The body-wall MNs are in the circuit but unmapped to any joint.

Selection detail: INs are `vnc_intrinsic` of any neuromere ranked by synapses to plus from the MN set (ties by bodyId). DNs (1314 candidates) and SNs (324 right and left ProLN proprioceptive and tactile candidates) are ranked by synapses onto MN+IN. Node order is role (DN, SN, IN, MN) then bodyId; edges are sorted by (post, pre) with a CSR `indptr` over rows = post.

## Signs

Rule (same as `extract_connectome_circuit.py`): acetylcholine +1, GABA -1, glutamate -1. `sign_code` makes the unknowns explicit: 0 = rule transmitter, 1 = other named transmitter (histamine, dopamine, ...; sign +1), 2 = `unclear` or missing (sign +1). Counts: DN 3 unclear, IN 3 unclear, SN 0, MN 60 unclear. No other-NT neurons were selected. Caveats:

- Glutamate as inhibitory is an assumption (GluCl), but it is not universal in the fly CNS.
- Fly MNs are glutamatergic at the NMJ, so the rule labels many MNs -1. This only affects the 77 MN out-edges, not the joint drive (which uses MN rates with the agonist/antagonist signs from the map).
- Unclear neurons default to excitatory; they are 6 of 3000 non-MN neurons.
- Edge sign is the presynaptic neuron's sign (`edge_sign`).

## MN to joint map (14 foreleg joints, `mn_joint_map.json`)

Joints are `joint_{LF,RF}{Coxa_yaw,Coxa,Coxa_roll,Femur,Femur_roll,Tibia,Tarsus1}` from `data/neuromechfly/model.json` (these are the actuated leg DOFs in `backend/physics.py`). L MNs drive LF, R MNs drive RF. Types match by exact `mn_type` string, across all nerves.

**No joint is a `fallback`**: contrary to expectation, the coxa is covered by T1 MNs leaving through ProAN/VProN (which are not in the ProLN set). But the coxa assignments (and Femur_roll) rest on muscle names only, so they are low confidence.

| joint DOF | agonist types | antagonist types | confidence |
|---|---|---|---|
| Tibia | Ti flexor, Acc. ti flexor, ltm1-tibia, ltm2-femur, ltm | Ti extensor | high |
| Tarsus1 | Ta depressor | Ta levator | high |
| Femur | Tr flexor, Acc. tr flexor | Tr extensor | medium |
| Femur_roll | Fe reductor | (none, one-sided) | low |
| Coxa | Tergopleural/Pleural promotor | Pleural remotor/abductor | low |
| Coxa_yaw | Pleural remotor/abductor | Sternal adductor | low |
| Coxa_roll | Sternal anterior rotator | Sternal posterior rotator | low |

Left out (action uncertain): Sternotrochanter MN (6), Tergotr. MN (8), and the 38 body-wall MNs. Two right ProLN MNs have no type annotation (bodyIds 1050340850 and 1052407897).

### Per-joint coverage (side-matched MN count / synapses received inside the circuit)

| joint | agonist MN / syn | antagonist MN / syn |
|---|---|---|
| LF Coxa_yaw | 2 / 14,070 | 1 / 878 |
| LF Coxa | 4 / 25,348 | 2 / 14,070 |
| LF Coxa_roll | 2 / 12,618 | 4 / 11,579 |
| LF Femur | 11 / 5,886 | 2 / 7,369 |
| LF Femur_roll | 4 / 9,576 | 0 / 0 |
| LF Tibia | 23 / 16,118 | 2 / 13,032 |
| LF Tarsus1 | 5 / 1,066 | 2 / 4,043 |
| RF Coxa_yaw | 2 / 10,124 | 1 / 599 |
| RF Coxa | 4 / 18,901 | 2 / 10,124 |
| RF Coxa_roll | 2 / 13,923 | 2 / 4,536 |
| RF Femur | 10 / 2,256 | 2 / 2,241 |
| RF Femur_roll | 6 / 8,508 | 0 / 0 |
| RF Tibia | 21 / 5,033 | 2 / 5,520 |
| RF Tarsus1 | 4 / 111 | 3 / 1,155 |

## Review items and risks

1. **Sign of each joint** relative to the MuJoCo axis is unverified. The gain `g_j` has free sign, so training can absorb it, but init at +1 could start anti-aligned.
2. **Thin antagonists**: Ti extensor has 2 MNs per side, Coxa_yaw antagonist has 1. Gain can compensate, but single-neuron pathways are fragile under ablation.
3. **RF Tarsus1 depressor receives only 111 in-circuit synapses**, versus 1,066 on the left. Tarsus drive may be weak on the right; consider raising IN k or lowering min_weight if it matters.
4. The `Pleural remotor/abductor MN` type serves two joints (Coxa antagonist and Coxa_yaw agonist), so those two joints are coupled.
5. The IN pool ignores neuromere (T2/T3 and abdominal INs can enter); the ranking is purely by MN synapses.
6. No proprioceptive-specific feedback from outside ProLN SNs (only foreleg ProLN afferents are used, left and right).
