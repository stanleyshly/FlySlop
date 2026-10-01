# FlySlop connectome coding curriculum: viability and implementation plan

Date: 29 September 2026. This plan replaces the previous `PLAN.md`. After approval, this file is written to `PLAN.md` and nothing else is implemented in this session. Each work package in §6 is sized for one **Sonnet 5.5 implementation subagent**. §7 says which packages can run in parallel.

## Context

`prompt.md` asks for four things:
1. Replace the "fake MLP" controller with the real MaleCNS connectome, using its actual foreleg motor neurons to drive leg position.
2. Decide whether MPS beats CPU.
3. Train through a ten-stage curriculum that goes from pressing single keys to autonomous code generation distilled from a frozen LLM (Gemma 4 E2B).
4. Run it all with one command.

The user confirmed three decisions:
- The code stages use **Python and SystemVerilog mixed from stage 6**.
- Each motor stage is **bootstrapped by imitation, then fine-tuned with PPO**.
- **This file replaces PLAN.md.**

### What exists today (verified by exploration)

**The learned actor does not move the legs.**
- `FlyTypingEnv` (`backend/fly_env.py:132`) takes an 8-D command: thorax velocity plus two claw XYZ targets.
- A scripted `OnlineGait` (`fly_env.py:59-94`) and IK (`backend/physics.py:362-404`) turn that command into all 42 joint targets.
- Motor neurons are only a latent vector feeding SB3's free linear `action_net`.

**The imitation is of a feedback expert, not an animation.**
- The current policy clones `expert_action()` (`fly_env.py:336-377`).
- That expert is a closed-loop scripted controller, not a replayed animation.

**The connectome actor is small and one-sided.**
- It lives in `training/connectome_policy.py`.
- It uses 360 neurons from the right foreleg only: 64 descending (DN), 64 sensory (SN), 192 interneurons (IN) and 40 motor (MN).
- Each step runs three unrolled `tanh` updates.
- Wiring and sign are fixed, synapse magnitudes are learned, and the encoders are free linear layers.

**PPO cannot train the connectome actor.**
- `training/train_ppo.py:158` hardcodes `PPO("MlpPolicy")`, so connectome runs are imitation-only.
- The critic is an MLP, which is fine because it is not used at inference.

**The text model is minimal.**
- The editor is an append-only string with Backspace (`backend/embodiment.py:53-56`), and there is no cursor.
- Arrow keys, Delete and Tab exist as physical MuJoCo keys but emit nothing.
- Shift is a one-shot latch (`backend/keyboard.py:131-138`), so selection is impossible.

**Throughput is low.**
- The environment runs about 400 steps/s per process, or about 770 steps/s across 8 processes.
- Typing takes about 50 ticks per character, so a learned policy types about 8 characters per second of wall clock per process.
- The scripted physics replay is about 100 characters/s.

**Data available locally.**
- MaleCNS v1.0 in `data/full_cns/`: 166.7k annotated neurons and 151.9M edges.
- 81 front-leg MNs exit via ProLN (41 left, 40 right, 12 types such as Ti flexor, Ta depressor and Fe reductor).
- The other 92 T1 MNs exit via other nerves, so coxa and body-wall control is partly outside ProLN.
- Around the right-foreleg MNs, the one-hop VNC neighbourhood is about 3.6k bodies.
- The VNC plus DN and ascending subgraph is 24k neurons and 4.0M edges, about 48 MB as sparse data.
- The lab checkout `lab-group40-fa25/` has 60 `.v`/`.sv` and 142 `.py` files. It has no licence, so it stays local.

**Hardware and tools.**
- The machine is an Apple M1 with 8 GB RAM, 8 cores and 75 GB free disk, running torch 2.14.
- **No local LLM runtime is installed.** There is no ollama, llama.cpp, mlx or transformers.

### MPS vs CPU (measured)

Recurrent matvec cost in ms per iteration:

| Neurons | CPU dense | CPU sparse CSR | MPS dense | MPS sparse |
|---|---|---|---|---|
| 360 (batch 1) | **0.006** | 0.047 | 0.015 | not supported |
| 5k (batch 64) | 8.3 | **1.9** | 2.6 | not supported |
| 20k (batch 64) | 125 | **4.2** | 73 | not supported |

**The verdict is CPU for everything with a connectome.** MPS cannot run `sparse_csr`, and it loses at small sizes because of dispatch overhead. The physics runs on CPU as well.

MPS or Metal is only useful for two things:
- running the LLM teacher, through llama.cpp or MLX rather than torch;
- optionally, the large dense token embedding and readout in stage 7.

P0 re-benchmarks the edge-list scatter backend (`index_add_`), because it supports autograd and also runs on MPS. Configs take `device: auto|cpu|mps`, and `auto` picks from a cached microbenchmark.

## 1. Viability verdict

| Stages | Verdict | Why |
|---|---|---|
| **Connectome drives legs** (motor-neuron layer) | **Viable, and this is the biggest engineering change** | We can map MN types to joints with antagonist pairs, for example Ti flexor against Ti extensor for the tibia. Coxa joints need non-ProLN T1 MNs or an explicitly labelled fallback. Imitation on joint angles derived from IK gives a strong bootstrap. |
| **1. Key control** | Viable | This is today's task with the action moved from claw targets to motor neurons. |
| **2+3. String and code-string typing** | Viable. **Merge them** into one transcription stage with a character-set curriculum. | Syntax characters are just more keys plus Shift. The cost is wall-clock time. |
| **4. Editing** | Viable, but it needs new editor infrastructure | We need a cursor, selection with held Shift, Delete, arrows and Tab, plus a planner that turns a diff into keystrokes. |
| **5. Error recovery** | Viable | Inject slips and edits into the buffer, re-plan from the observed buffer, and train on recovery. |
| **6. LLM oracle** | Viable offline, with risk | Gemma 4 E2B at 4-bit needs about 3 GB, so it must run alone, not alongside 8 MuJoCo workers. It runs at an estimated 15–30 tokens/s on the M1, so 20k pairs is roughly 12–24 hours. Small models write weak SystemVerilog, so outputs must pass pyslang or execution tests before we keep them. **Confirm the exact checkpoint, licence and runtime in P0.** |
| **7. Distillation** | Viable as an experiment | Train a recurrent connectome student with next-token loss, then scheduled sampling from A to C. The token embedding and readout will hold more parameters than the wiring, so **ablation controls are mandatory** before claiming anything. |
| **8. Autonomous short programs** | Plausible for templated micro-tasks | Expect success on problems like "return 5", "double x" and "max of two" that are in or near the training distribution. Expect poor generalisation to novel semantics. |
| **9. Longer programs** | **Research risk, likely to plateau** | Memory in a few thousand tanh units with limited recurrent steps is the constraint. Multi-function programs are unlikely to work. |
| **10. Self-correction** | Viable as infrastructure, uncertain as learning | Writing, running tests, observing and editing is easy to build. Learning to repair from test output is hard, so start with the diff-oracle editing policy from stages 4–5 conditioned on a failure signal. |

**Key design decision that makes the plan viable:** training is split into a **physical motor level** and a **symbolic level**.
- **Physical, stages 1–5.** A VNC "typing connectome" learns to press any requested key, in MuJoCo.
- **Symbolic, stages 4–10.** A brain "thinker connectome" decides the next action token. That is a code token or an edit token such as `<LEFT>`, `<BS>` or `<SEL>`. It trains on buffer states with no physics, which is roughly 10⁴ times faster.
- **Physical typing appears only in evaluation and demos** of the upper stages.

This is the split the user described ("connectome: what comes next; typing system: how to type it"). Descending neurons form the biological interface between the two levels.

**Realistic total compute on the M1** is about 1–3 days wall clock, mostly for stages 1–5 PPO and stage 6 teacher generation. The user runs it offline. Claude only smoke-tests; see [[offline-training-handoff]].

## 2. Architecture

```text
prompt / target / buffer+cursor / test output
        │  (tokenised, symbolic)
        ▼
THINKER connectome (brain subgraph: visual_projection + cb_intrinsic sample → DN)
        │  next action token  (code token | edit token | <RUN> | <EOS>)
        ▼
token → characters → key-command sequence   (deterministic; Shift/arrows expanded)
        │  one key id at a time
        ▼
TYPING connectome (bilateral T1 circuit: DN + SN + VNC IN + 81 foreleg MNs)
        │  MN rates → antagonist pooling → 14 foreleg joint targets
        ▼
MuJoCo NeuroMechFly (4 other legs keep OnlineGait; thorax still mocap)
        │  key contacts
        ▼
Editor (text, cursor, selection) → monitor, judge, observation
```

### Typing connectome (P1 and P2)

**Neurons.**
- All 81 ProLN foreleg MNs, plus the T1 non-ProLN MNs that map to coxa muscles.
- The one-hop VNC interneurons of those MNs, taking the top-k by synapse count, with k set in config (default about 2–3k).
- DNs that synapse onto the set (about 150–300).
- Foreleg proprioceptive and tactile SNs (about 100–200).
- The target is about 3–4k neurons and about 100k edges. Sparse CPU cost is about 1 ms per step, which is below physics cost.

**Signs.** The existing neurotransmitter rule is kept: ACh is excitatory, GABA and glutamate are inhibitory. The assumption is documented.

**Inputs.**
- Key command: a learned key-id embedding goes into the DNs. The existing next-key offsets are an optional, state-assisted flag, on by default for stage 1.
- Proprioception and touch go into the SNs.

**Outputs.**
- There is **no free linear readout.**
- Each joint target is `rest + g_j·(Σ agonist MN rate − Σ antagonist MN rate)`, with a scalar gain and offset learned per joint. That is 28 parameters.
- The MN-to-joint table is written in `data/connectome/mn_joint_map.json` and reviewed. Any joint without MN coverage is marked `fallback` and reported.

**Body velocity** is handled first by a small DN-rate readout, labelled as such. MN-driven walking with T2 and T3 MNs is a later extension.

**Bootstrap.** Expert claw targets go through `LegIK` to give joint angles. Imitation is an MSE on the joint angles, then PPO optimises the real typing reward.

### Thinker connectome (P1 and P6)

**Subgraph.**
- The input population is a sample of `visual_projection` neurons, which frames it as the fly reading the monitor.
- The sample is connected through `cb_intrinsic` neurons to DNs.
- Size is set in config at 5k–20k neurons and about 0.5–2M edges, run as CPU sparse.
- Tokens are embedded into the input neurons. Each token takes K recurrent micro-steps, default 4. The readout comes from the DNs.

**Tokenizer.** A small BPE trained on the mixed Python+SV corpus, with vocabulary of 1–2k plus edit and control tokens. We do not use Gemma's vocabulary, because the embedding and readout would swamp the connectome.

**Controls, all at equal trainable parameters and budget:**
- a dense GRU;
- shuffled wiring;
- random sparse wiring;
- readout-only, with the connectome frozen at random magnitudes.

Any claim that the connectome "absorbed coding" requires beating shuffled and random-sparse wiring with a bootstrap CI. That is the same rule as `training/connectome_experiment.py`.

## 3. Curriculum stages and gates

Each stage runs imitation or supervised training first, then its main trainer, then a held-out gate. The orchestrator moves on only when the gate passes. When the budget runs out it stops and reports, unless `--advance-on-budget` is set.

| # | Stage | Level | Trainer | Held-out gate (initial, freeze before full runs) |
|---|---|---|---|---|
| 1 | Key control: 8 keys, then 26, then the full keyboard | physical | BC→PPO | ≥95% correct single-key presses, ≤1% unintended characters |
| 2 | Transcription: words, then code strings with symbols, Shift, Enter and Tab (merges the prompt's stages 2 and 3) | physical | BC→PPO, target pools switched by character set and length | ≥90% exact on held-out 3–8 character tokens, and ≥70% exact on held-out 1–2 line code strings |
| 3 | Editing: arrows, Backspace, Delete, selection, insert and replace | physical + symbolic | Physical: BC→PPO on edit-key commands. Thinker: supervised on (target, buffer, cursor) → diff-oracle action | Physical: ≥95% correct edit keys. Symbolic: ≥95% of edits reach the target within 1.5× the oracle's keystrokes |
| 4 | Error recovery: injected wrong, missing and duplicated characters, bad indents and cursor moves | physical + symbolic | Physical slips injected by occasionally corrupting the key command. Thinker trained on corrupted buffers | ≥90% recovery to exact target, in physical evaluation on 50 held-out episodes |
| 5 | LLM oracle: collect data | offline | Gemma generates (prompt, code) pairs that must pass filters. A sample is typed physically for a demo | ≥10k filtered pairs, measured pass rate recorded, deduplicated train/test split by source and similarity |
| 6 | Distillation A→B→C | symbolic | Teacher forcing, then scheduled sampling 0→25→50→75→~100% student tokens, then student-only | Held-out: token accuracy, then parse or execution pass rate. Must beat the shuffled and random-sparse controls |
| 7 | Autonomous micro-programs | symbolic + physical eval | Evaluation only, plus fine-tuning on failures | Pass@1 on a fixed micro-suite (about 50 Python and 30 SV tasks) with tests, typed physically on a subset |
| 8 | Longer programs, one to five lines, then branches, loops and multiple functions | symbolic | Continue distillation with a length curriculum | Report pass rate against length. Advance only while above the configured floor |
| 9 | Self-correction: write, run, observe, edit, run | symbolic + physical eval | Test output tokenised into the thinker context. Starts from the stage 3–4 edit policy, then REINFORCE or rejection-sampling fine-tune on test-pass reward | Improvement in pass@1 after up to 3 repair rounds compared with no repair |

The prompt's stages are renumbered here: stages 2 and 3 are merged, so its 10 stages become 9. The prompt's stage 1 becomes stage 1 here. Its stage 6 (oracle) becomes stage 5, and its stages 7–10 become stages 6–9.

## 4. Data sources

**Strings and code strings (stage 2).**
- Tokens and lines from the authored corpus and from the local lab checkout, extracted locally and git-ignored.
- MBPP (CC-BY-4.0) and HumanEval (MIT) solutions.
- VerilogEval and RTLLM, **after verifying their licences in P5**.

**RTL prompt synthesis (stage 5).**
1. Extract modules and always-blocks from the lab `.v`/`.sv` files with pyslang, and Python functions from the lab `.py` files with `ast`.
2. Gemma writes a natural-language prompt for each extract.
3. The prompt goes back to Gemma for fresh code.
4. Keep the pair only if the code passes the filters: pyslang parse and elaboration for SV, and execution tests for Python where tests exist.
5. Seed the process with MBPP and VerilogEval prompts that already have tests.

**Rights.** All lab-derived pairs stay in git-ignored `data/private/`, the same rule as in `data/corpus/README.md`.

## 5. Critical files

**New:**
- `training/curriculum.py`: the orchestrator.
- `training/curriculum.json`
- `backend/connectome/runtime.py`: sparse recurrent layer and ablations.
- `backend/connectome/typing_circuit.py` and `backend/connectome/thinker.py`
- `scripts/extract_typing_circuit.py`, `scripts/extract_thinker_circuit.py`
- `data/connectome/mn_joint_map.json`
- `backend/editor.py`
- `backend/keyplan.py`: token to keys, and diff-oracle edits.
- `training/teacher.py`: runs Gemma through llama.cpp, MLX or Ollama.
- `training/datasets.py`, `training/tokenizer.py`, `training/distill.py`
- `backend/exec_judge.py`: sandboxed Python tests, plus SV with iverilog or Verilator.
- `scripts/bench_device.py`

**Modified:**
- `backend/fly_env.py`: add an `action_mode="mn"` action with 14 joints plus a velocity readout, a key-command input, and editor observation.
- `backend/physics.py` and `backend/keyboard.py`: held modifiers, and outputs for arrows, Delete and Tab.
- `backend/embodiment.py`: `apply_key` becomes editor-based, and `replay_text` must still verify contacts.
- `training/connectome_policy.py`: replace the dense-mask actor with runtime.py.
- `training/train_ppo.py`: fix the `MlpPolicy` hardcode and use `make_model`.
- `training/imitate.py`: joint-space targets and device selection.
- `training/common.py`: extend `code_hash` to cover the connectome files.
- `backend/policy_replay.py` and `web/src/main.js`: brain panel for the new circuits.
- `docs/connectome.md`, `docs/training.md`, `docs/results.md`, `README.md`

**Reuse:**
- `CurriculumAndEval` (`train_ppo.py:36-68`) for stage switching.
- `demo_worker` and `fit` (DART-style noisy demos) in `imitate.py`.
- `LegIK`, `OnlineGait`, and the `expert_action` logic.
- The seven-variant ablation and bootstrap CI in `connectome_experiment.py`.
- `judge.py` (pyslang).
- `code_hash` and run manifests (`common.py`).
- The token splits in `tokens.py`.

## 6. Work packages for Sonnet 5.5 subagents

Every package ends with the same handoff: files changed, commands run, measured numbers, remaining risks, and unit tests added under `tests/`. No package runs long training. Smoke tests stay under about 2 minutes.

- **P0 — Benchmarks and contracts (first, serial).**
  - Create `scripts/bench_device.py`, benchmarking dense, CSR and `index_add_` edge-scatter backends on CPU and MPS, forward and backward, at 360, 4k and 20k neurons.
  - Pin the connectome runtime backend.
  - Install one LLM runtime, recommended llama.cpp or MLX.
  - Verify that the Gemma 4 E2B checkpoint exists, check its licence, and measure RAM use and tokens/s on the M1.
  - Freeze the key-command, action-token, and editor-event schemas in `docs/architecture.md`.
  - **Gate:** a benchmark table, a working `teacher.generate("…")`, and the schemas committed to docs.
- **P1 — Circuit extraction.**
  - Build the typing and thinker extractors (above) from the feather files. Avoid peak-RAM blowups on 8 GB by using pyarrow filtered reads and cached `.npz` output.
  - Write the MN-to-joint map with a review note.
  - **Gate:** deterministic hashes, a coverage report per joint, and neuron and edge counts recorded.
- **P2 — Connectome runtime and SB3 integration.**
  - `runtime.py` covers sign × exp(log_mag) × mask, K micro-steps, and ablation variants.
  - Build `ConnectomeActorCriticPolicy` with the MLP critic kept, fix `train_ppo.py`, and add device handling.
  - **Gate:** the imitation and PPO smoke run works with `--policy connectome`, the checkpoint reloads, and `policy_replay` streams activity.
- **P3 — MN-driven legs env** (needs P1 and P2).
  - Add `action_mode="mn"` in `FlyTypingEnv`, joint-space expert labels from IK, and a key-id command input.
  - **Gate:** imitation smoke on 8 keys types at least some correct characters. Scripted-expert parity is unchanged in the old mode.
- **P4 — Editor, keyboard and key planner** (parallel with P1).
  - Build `backend/editor.py` (text, cursor, selection) and the held-Shift and modifier state machine.
  - Add arrows, Delete and Tab, and the replay contract with a cursor.
  - Build `keyplan.py`, which turns tokens into keys and diffs into minimal edit keystrokes, plus the error injector.
  - Add the viewer cursor rendering.
  - **Gate:** the physics gate is re-met, `replay_text` still rejects contactless events, and property tests show that applying `oracle_edits(buffer, target)` gives `target`.
- **P5 — Data and teacher pipeline** (parallel, after P0).
  - Build dataset loaders, with licences recorded in `THIRD_PARTY.md`.
  - Add lab extraction, prompt synthesis, and filtered generation with resumable JSONL.
  - Build the BPE tokenizer, and split by source and similarity.
  - Build `exec_judge.py`: Python runs in a subprocess with a timeout and no network, and SV uses pyslang plus iverilog testbenches where available.
  - **Gate:** 200 filtered pairs generated in the smoke run, with the filter pass rate reported.
- **P6 — Thinker and distillation** (needs P1 and P5).
  - Build `thinker.py` and `distill.py` with a teacher-forcing then scheduled-sampling schedule.
  - Add the edit-token supervised trainer for stages 3–4 and the four controls.
  - **Gate:** overfits 100 pairs, the schedule is logged, and the control runs share one config.
- **P7 — Evaluation and self-correction loop** (needs P4, P5 and P6).
  - Build the micro-suite with tests, the loop env (write, run, observe, edit), and a physical-eval harness that types the thinker's output through the typing connectome.
  - **Gate:** the oracle loop solves the suite, and the loop terminates on timeouts.
- **P8 — One-command orchestrator** (skeleton early, finished last).
  - Command: `uv run --extra training python -m training.curriculum --config training/curriculum.json [--smoke] [--resume RUN] [--from-stage N]`.
  - It runs a state machine persisted at `runs/curriculum/<ts>/state.json`, with per-stage manifests (code, data and circuit hashes), gates, budget limits and resumption.
  - It runs the teacher stage alone so the teacher does not compete for RAM with the MuJoCo workers.
  - **Gate:** `--smoke` runs every stage end to end in under about 15 minutes, and a killed run resumes.
- **P9 — Docs and viewer integration.**
  - Rewrite `docs/results.md` status, `training.md` and `connectome.md`.
  - Label the brain panel for the typing and thinker circuits, keeping modelled activity distinct from measured wiring.
  - List the limits.

## 7. Order and parallelism

1. P0.
2. P1, P4 and P5 in parallel. The P8 skeleton can start here too.
3. P2, after P1.
4. P3 (after P2) and P6 (after P1 and P5) in parallel.
5. P7.
6. P8 finish, then P9.

Use separate git worktrees for the parallel packages. The integration owner rejects any package that breaks deterministic replay or the contact-to-character chain.

## 8. Risks to track

- **Coxa joints.** Motor neurons may not cover the coxa joints cleanly. Use the fallback and report it; this is not a failure.
- **PPO on MN outputs.** PPO may destabilise the imitation policy on MN outputs. Mitigate with KL or behaviour-cloning regularisation and a low learning rate. The current PPO config already uses a low learning rate.
- **Weak teacher on SV.** Gemma E2B's SV quality could make the filtered yield too low. The fallback is to seed from VerilogEval or RTLLM solutions and use Gemma only for paraphrasing prompts. A larger teacher, such as Gemma E4B, will not fit alongside anything else on 8 GB.
- **Parameter imbalance.** Embedding and readout parameters dominate the thinker, so claims must come from the ablations.
- **8 GB RAM.** Load circuits from cached `.npz`, never the 1 GB feather at train time. Run the teacher alone.
- **Unlicensed lab code.** It stays local; never commit derived pairs.

## 9. Verification

- `uv run python -m unittest discover -s tests -v`, extended with tests for the editor, key planner, MN map, runtime ablations, exec judge and orchestrator state.
- `uv run python scripts/evaluate_physics_gate.py`: re-meet the scripted gate after the P4 keyboard changes.
- `uv run --extra training python -m training.curriculum --config training/curriculum.json --smoke`: every stage runs with tiny budgets and writes gate reports.
- `uv run python scripts/bench_device.py`: records the CPU vs MPS decision.
- `uv run python -m backend.server --policy runs/curriculum/<ts>/stage1/best_model.zip`: watch a motor-neuron-driven key press. The brain panel shows the typing circuit.
- The full run is done offline by the user with the same command without `--smoke`. Update `docs/results.md` from the stage reports.
