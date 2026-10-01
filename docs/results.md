# Current measured results

## Connectome curriculum status (PLAN.md P0-P8, 2026-09-30)

**Nothing in the connectome curriculum has been fully trained.** All nine
stages are implemented and the code paths run, but only smoke runs exist
(relaxed gates, tiny budgets, minutes each). No gate of PLAN §5.3 has been met
by a learned connectome policy, and no connectome benefit is claimed. See
[Known limitations](#known-limitations) before reading any number below.

| Piece | State |
| --- | --- |
| Typing circuit (3,173 neurons, 98.5k edges), thinker circuits (5k/537k and 17k/1.05M) | extracted and hash-checked |
| `SparseRecurrent` runtime (csr/index_add/dense; real/shuffled/random_sparse/frozen) | implemented, benchmarked |
| MN legs (`action_mode="mn"`, 16-D, 28-parameter antagonist readout) | implemented; BC smoke typed 6/24 (proof of life only) |
| Editor, keyplan oracle (600/600 exact), error injection, chords | implemented; physics gate unchanged |
| Data (MBPP, HumanEval, VerilogEval, RTLLM, authored, private) and exec judge | implemented |
| Oracle generation (`oracle_gen`) | only the stand-in (Qwen2.5-Coder-0.5B) smoke has run |
| Thinker distill with gated schedule | GRU, 100 pairs: reached phase C, 0.949 gen token acc, 0.89 pass; not run on connectome kinds |
| Microsuite (51 Python + 38 SV) | oracle 100%, null 0%, repair +0.57 |
| Curriculum orchestrator, stages 1-9 | stages 1-4 and 6-9 smoke-ran; **stage 5 smoke has not run** (needs a ~2 GB cap) |

The sections below this table predate the curriculum. They describe the
earlier 8-D, 360-neuron setup and the scripted-body gate, and are kept as
measured at the time.

Measured on 2026-09-26 on an Apple-silicon laptop (CPU only). Short runs are
engineering checks. The PLAN §5.3 learned-task and connectome gates need the
full multi-seed runs in [training.md](training.md), which have not been run
yet.

## Corpus, oracle, kinematic demo

| Check | Result |
| --- | --- |
| Corpus digest, parse, standalone elaboration | 24/24 targets pass `scripts/validate_corpus.py` |
| Keyboard oracle through contact path | 24/24 exact, replay reproduces text (oracle gate met) |
| No character without contact; one per press; Shift is a physical press | unit tests pass |

## Physical body (MuJoCo), scripted controller: gate met

`scripts/evaluate_physics_gate.py`, 3 seeds × ~200 random presses over the corpus alphabet ([physics_gate.json](physics_gate.json)):

| Metric | Result | Gate |
| --- | --- | --- |
| Intended presses producing a contact on the intended key | 601/601 (100%) | ≥ 95% |
| Unintended characters | 0/601 onsets (0%) | ≤ 1% |
| Exact text per seed | 3/3 | — |
| Replay integrity | 3/3 | required |
| Uncredited key actuations (no character emitted) | 37 | reported |

Re-measured on 2026-09-27 on the fly-sized laptop (0.3 mm key pitch, life-size
fly), with both forelegs typing. Before the rescale: 1/602 unintended, 24
uncredited. On the 24 corpus targets, the scripted planner types every target
exactly. Of 7,673 presses, LF made 4,161 and RF 3,512; 78% needed no body
motion, 11% a thorax lean over planted feet, and 11% a walk.

The body is NeuroMechFly v2 legs driven by 42 position actuators, with claws
pushing spring-loaded keys. The thorax is carried by mocap and gravity is
off (see [physics.md](physics.md)). `fly_demo` is typed exactly in physics
(`data/replays/fly_demo.physics.json`).

## Learning, fly body (`FlyTypingEnv`)

**Stale:** these numbers are from the earlier 10×-fly, right-foreleg-only
environment (5-D action, 22-D observation). The env is now two-legged (8-D
action, 33-D observation) on the fly-sized laptop, so the checkpoints and
numbers below no longer apply. Re-run [training.md](training.md). After the
change, the scripted expert typed 40/40 episodes (8 targets × 5 seeds;
LF 143, RF 157 keys).

| Controller | Held-out tokens exact | Notes |
| --- | --- | --- |
| Random actions | 0/20 | negative mean return after the reward fixes |
| Scripted expert (baseline) | 42/42 (train 139/142) | train failures are walking legs landing on keys |
| Imitation, MLP actor (240 demos, 40 epochs, 77 s) | 30/30 | trained on train tokens only |
| Imitation + PPO fine-tune 150k steps | 30/30 at the end | no collapse; ~770 steps/s on 8 processes |
| Imitation, connectome actor (same budget) | 16/20 | fewer effective connections |

The imitation MLP policy, which never saw whitespace, types all of
`fly_demo` (65 characters with spaces, newlines, `_`, `;`, `'`). The result
passes exact, syntax, and elaboration checks. The connectome-actor policy
stalls in the indentation. These are single-seed, short-budget runs, not the
§5.3 gate (5 seeds × 100 held-out episodes).

The 150k PPO check ran before the reward fixes in [training.md](training.md).
The fly-body PPO smoke test was re-run afterwards.

## Learning, point foot (`KeyboardTypingEnv`)

PPO from scratch, 400k steps, single seed: held-out exact went 0 → 1.00
over 100 episodes in about 40 s, at ~10k steps/s. The random baseline is 0.

## Connectome

A 360-neuron MaleCNS v1.0 right-foreleg circuit was extracted (5,549 edges,
92,900 synapses). The paired ablation pipeline ran as a smoke test only.
**No connectome benefit is claimed** (see [connectome.md](connectome.md)).

## Not established

- Pixel-only transcription: the target key location is in the observation.
- Weight-bearing locomotion or biologically plausible key forces.
- Held-out full-module transcription beyond `fly_demo`, or a keyboard-layout
  perturbation test.
- Any causal contribution of connectome wiring.

## Known limitations

- **Fn chords are unreachable.** The Fn key cannot be held by a foreleg
  geometrically, so Fn combinations use a sticky-Fn latch. This deviates from
  contract C2 (held chords). Shifted symbols `! # + @ _ ~` also fall back to the
  Shift latch rather than a held two-leg chord.
- **Unverified MN joint signs.** The signs for Coxa_roll, Femur, and Femur_roll
  are not verified against the body ([data/connectome/mn_joint_sign_check.json](../data/connectome/mn_joint_sign_check.json)).
- **Stateless actor.** The per-step connectome actor carries no recurrent state
  across env steps, so it cannot remember what it was doing.
- **Speed.** The connectome thinker is about 100x slower than the matched GRU. One
  thinker step at B=32, T=128 takes about 3.5 s on a quiet machine. The ~4x
  threaded speedup was measured once, under load.
- **Shuffled control propagates more strongly than real wiring.** A shuffled
  control is therefore not a clean null; interpret real-vs-shuffled with care.
- **Stage 5 data volume.** Stage 5 needs about 10k oracle pairs. At Gemma
  throughput (~26 tok/s) that is about 87 h. The fallback in PLAN §8 has to be
  chosen before a full run.
- **Gated schedule untested on connectome kinds.** The defaults (`mode=gated`,
  `mix_mode=parallel`, `tf_lambda=0.5`) were tuned on a GRU only. Sequential
  mixing collapsed at p >= 0.75.
- **State-assisted observation, kinematic thorax, game-scale key forces** still
  apply (see "Not established").
