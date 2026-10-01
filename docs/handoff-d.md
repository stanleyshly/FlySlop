# Package D handoff: contact-gated learning scaffold

## Deliverables

- `backend/env.py` provides `KeyboardReachEnv`, an optional Gymnasium environment. Its absolute xyz action sets one scaled point foot. It sends every pose through `ContactKeyboard`; a character can only enter the buffer after the existing press/release contact state machine emits it. The observation is explicitly state-assisted and includes the next target key center and shift requirement.
- `training/config.json` records seed, target set, observation/action contract, null algorithm/checkpoint, and the unmet gate.
- `training/oracle_demo.py` writes the existing deterministic scripted embodiment replay. `training/evaluate_oracle.py` drives the Gymnasium action interface with deterministic target-to-key lookup and prints per-target results. Both are oracle/scripted demonstrations, not learned policy measurements.
- Gymnasium and NumPy are optional under `uv sync --extra training`.

## Gate and claim status

The repository currently has a kinematic point-foot contact proxy, not an articulated physical body with calibrated foot/key contacts. Package D therefore stops at an environment and reproducible oracle scaffold. PPO was not trained, no policy checkpoint exists, and there is no learning threshold result. Do not describe oracle exact-match performance as learning. The action is state-assisted, so it also does not establish pixel-only transcription or biological plausibility.

The environment's compact target state is for motor-control scaffolding. A research evaluation must prevent target leakage beyond the declared visible target condition, use held-out strings/layouts, and compare against random and scripted baselines. Training stays gated until Package C satisfies the articulated body/contact gate and the prior keyboard oracle and scripted-body thresholds.

## Commands

From the repository root:

```sh
uv run --extra training python -m unittest tests.test_training_scaffold tests.test_contact_replay
uv run --extra training python -m training.oracle_demo fly --output /tmp/fly-oracle.json
uv run --extra training python -m training.evaluate_oracle fly abc sv --seed 0
```

The test checks replay reconstruction and, when the optional dependency is present, checks that the environment requires contact/debounce before emitting. The scripts print mode/claim labels in their output. `training/config.json` is configuration metadata, not evidence of a trained run or checkpoint; a future checkpoint manifest should add code commit, dataset hash, full config hash, seed, and train/eval split.

## Update 2026-09-26: physics body and learning

This scaffold was superseded the same day. There is now a MuJoCo leg system
(`backend/physics.py`, whose scripted-body gate is met) and a fly-body
environment (`backend/fly_env.py`). Training goes imitation → PPO
(`training/imitate.py`, `training/train_ppo.py`, config `training/ppo_fly.json`).
Evaluation is `training/evaluate_policy.py`, and the five-seed sweep is
`training/run_fly_seeds.sh`. See [training.md](training.md) and
[results.md](results.md). The point-foot `KeyboardTypingEnv` remains as a
fast control task.
