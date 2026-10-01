# Training

**Status.** The connectome curriculum (stages 1-9, below) is implemented but
**not fully trained**; only smoke runs exist. The fly-body PPO sections further
down describe the older single-environment recipe.

## RAM cap

Every job runs under a configurable cap (`backend/memory_budget.py`). The
precedence is `--ram-gb` > env `FLYSLOP_MAX_RAM_GB` > config `ram_gb`, and the
default is **4 GB**. Worker counts are sized from the cap, and a watchdog stops
a job that exceeds it.

## Connectome curriculum (`training/curriculum.py`)

`training/curriculum.json` configures nine stages, each with a budget and a gate
(`--smoke` relaxes the gates). Implementations are in `training/stages/`.

| Stage | What | Notes |
| --- | --- | --- |
| 1 keys | BC then PPO over growing key pools (8 home-row, 26 letters, all) | gate: >=95% single-key presses correct, <=1% unintended |
| 2 transcription | words, then code of 16/32/64/128 characters and multi-line | warm-started from stage 1 |
| 3 editing | Backspace/arrows, selection (editor + keyplan) | key-queue BC then PPO |
| 4 recovery | slipped-then-corrected key strings (`error_inject`) | |
| 5 oracle data | `training.oracle_gen` with the Gemma teacher (exclusive) | ~10k pairs; ~87 h at Gemma speed |
| 6 distill | thinker distilled A (teacher forcing), B (scheduled sampling), C (student only); gated schedule | `training/thinker.json` |
| 7 micro-programs | pass@1 on the microsuite plus physical typing of the policy output | |
| 8 longer programs | length curriculum continuing stage 6 | |
| 9 self-correction | write, run, observe, edit, run (`selfcorrect_train.py`) | |

Stage 5 smoke has not run (needs a ~2 GB cap). Stages 1-4 and 6-9 smoke-ran
with relaxed gates. Known limits: [results.md](results.md#known-limitations).

Commands (the user runs the long ones offline; Claude only sets up and smoke tests):

```sh
# Gemma gate run (~1.5 h, nothing else running)
FLYSLOP_MAX_RAM_GB=4 uv run --extra teacher --extra training python -m training.oracle_gen \
    --run gemma-gate200 --limit 200 --time-budget-s 7200 --seed 0

# full curriculum (add --resume RUN to continue a run)
uv run --extra training python -m training.curriculum --config training/curriculum.json \
    --advance-on-budget --ram-gb 4

# smoke: every stage with relaxed gates (stage 5 needs FLYSLOP_MAX_RAM_GB=2)
uv run --extra training python -m training.curriculum --config training/curriculum.json --smoke --advance-on-budget

# connectome thinker overfit check
uv run --extra training python -m training.distill --overfit 100 --kind real --k-steps 2 \
    --steps 400 --out runs/distill/overfit_real
```

Readers of a distill `schedule.jsonl` must skip rows that have an `"event"` key.

## Tokenizer (`training/tokenizer.py`, contract C3)

A pure-Python BPE (1024 tokens) over printable ASCII. Id layout: 0-15 fixed
action/edit/special tokens, 16-31 reserved, 32-126 printable ASCII, 127+ merges.
**`SEP` = 16 and the edit markers 17-19 sit inside the reserved range (16-31)**,
so they must not be reused for anything else. Newline and tab are emitted as
`<ENTER>` (4) and `<TAB>` (5).

There are two learning environments. Both emit characters only through a
contact state machine, and every evaluated episode is checked with
`backend.embodiment.replay_text`. Both are **state assisted**: the observation
contains the offset to the next physical key (Shift before a shifted
character). They measure learned motor control of typing, not reading.

| | Point foot (`training/ppo.json`) | Fly body (`training/ppo_fly.json`) |
| --- | --- | --- |
| Environment | `backend.env.KeyboardTypingEnv` | `backend.fly_env.FlyTypingEnv` |
| Body | one kinematic point | NeuroMechFly v2 legs in MuJoCo (`backend/physics.py`) |
| Key contact | foot z below a threshold over a key | spring-loaded key pushed past `press_depth` while a claw touches it |
| Action | normalized `(dx, dy, dz)` foot step | thorax velocity `(vx, vy)` + right-foreleg claw target `(rx, ry, rz)` |
| Recipe | PPO from scratch | imitation of the scripted expert, then PPO fine-tuning |
| Throughput (M-series CPU) | ~10k steps/s | ~400 steps/s per process, ~770 steps/s with 8 processes incl. eval |

## Fly body environment

`FlyTypingEnv` follows the control hierarchy in PLAN §5.1. The policy
commands a walking velocity. An online gait moves the four middle and hind
legs in diagonal pairs. Both forelegs type: the policy sets each claw target
inside a reach box around that leg's press anchor. The action is 8-D:
`vx, vy`, then `rx, ry, rz` for RF, then `lx, ly, lz` for LF. The observation
gives the next-key offset and claw state for each foreleg, plus `key_side`,
which says whose side of the body the key is on. IK converts all six claw targets to 42 joint targets. MuJoCo position
actuators then drive the joints, and claw spheres push keys. A walking leg
that pushes a key types it like any other key. Such a character counts as a
wrong keypress and ends the episode.

Physics constants are frozen in `backend.physics.PhysicsParams`. The thorax is
carried kinematically by a mocap body, so the legs do not bear weight, and
gravity is off. Key stiffness, preload, and travel are game-scale values
chosen so a leg can press a key on the fly-sized laptop. See [physics.md](physics.md).

`FlyTypingEnv.expert_action()` is the scripted baseline. It uses only
observable state, so it can be cloned. The key goes to the foreleg on its
side of the body. The expert walks until the key is in that leg's reach, aims
the claw, strikes when the claw is well inside the keycap, and lifts in place
after a press before anything moves. The other foreleg hovers.

## Rewards

The weights are frozen in each config: +1 for a correct character, −1 for a
wrong one, ±0.5 for a needed or unneeded Shift, +2 for completion, and a small
per-tick cost. Shaping adds `Φ(s′) − Φ(s)`, where Φ is higher closer to the
next key. Over the key, Φ rewards lowering when armed and lifting after a
press. Φ′ is set to 0 only on success. Two bugs were found and fixed:

- Zeroing Φ′ after an error paid a bonus for typos.
- Discounted shaping `γΦ′ − Φ` with Φ < 0 paid the policy for standing still.

With the fixed reward, random actions score negative return and the scripted
controller scores clearly positive.

## Data splits

Targets are 3–8-character whitespace tokens from the corpus (`training/tokens.py`).
Train tokens come from `train`-split files. Held-out tokens come from
`test`-split files, minus any token also seen in training (142 train, 42 held
out). The curriculum goes single characters → train tokens ≤ 4 characters → all
train tokens.

## Commands

```sh
uv sync --extra training --extra connectome

# smoke tests (seconds each; not expected to reach the gate)
uv run --extra training python -m training.train_ppo --smoke                            # point foot
uv run --extra training python -m training.imitate --config training/ppo_fly.json \
    --episodes 16 --epochs 5 --eval-episodes 4 --out runs/bc/smoke
uv run --extra training python -m training.train_ppo --config training/ppo_fly.json --smoke \
    --init-from runs/bc/smoke/bc_model.zip                                              # fly body

# one full fly-body seed: imitation (~2 min) then 5M PPO steps (~2 h on 8 processes)
uv run --extra training python -m training.imitate --config training/ppo_fly.json --seed 0 --out runs/bc/seed0
uv run --extra training python -m training.train_ppo --config training/ppo_fly.json --seed 0 \
    --init-from runs/bc/seed0/bc_model.zip

# everything the gates need: 5 seeds (imitation + PPO), evaluation, connectome ablation
training/run_fly_seeds.sh                 # add SCRATCH=1 for a PPO-from-scratch arm
training/run_seeds.sh                     # point-foot 5-seed sweep

# evaluate runs, view curves
uv run --extra training python -m training.evaluate_policy runs/fly/<sweep>/ppo-seed* --episodes 100 --output report.json
uv run --extra training tensorboard --logdir runs
```

Every run directory (git-ignored under `runs/`) holds:

- `manifest.json`: config, config/code/dataset SHA-256, git commit, seed, package versions, and `init_from`
- `splits.json`
- `eval.jsonl` for held-out progress
- checkpoints, `best_model.zip`, and `final_model.zip`
- TensorBoard logs

`evaluate_policy` reports held-out and train exact match and CER per run,
plus the median and range across runs. It also runs random-action and
scripted baselines on the same episodes and checks the gate (≥ 5 non-smoke
seeds, all ≥ 0.9 held-out exact).

`best_model.zip` is selected on held-out exact match during training, which
is optimistic. For the frozen gate number, evaluate `final_model.zip` (or add a
validation split) with `--model final_model.zip`.

## Watching a learned policy

```sh
uv run --extra training python -m backend.server --policy runs/bc/seed0/bc_model.zip
```

Choose **Learned policy · physics** in the source menu. If the checkpoint has
a connectome actor, the brain panel shows its 360 MaleCNS neurons at their
soma positions, coloured by modelled activity.

## Claim status

See [results.md](results.md). A passing learned-task gate would show one
thing: a policy learned contact-gated walking-and-pressing to type unseen
token strings with an articulated, physically simulated leg system, given
the key location. It would not show pixel transcription, weight-bearing
locomotion, or a connectome effect (see [connectome.md](connectome.md)).
