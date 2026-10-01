# FlySlop

FlySlop is a local research prototype: a NeuroMechFly types on a fly-sized laptop keyboard, with a replay viewer and a connectome-based coding curriculum.

## Architecture

- **`backend/`** owns MuJoCo simulation, leg control, physical key contacts, the editor, replay integrity, and Python/SystemVerilog judging. Characters come from key contacts; the browser renders the recorded output.
- **`backend/connectome/`** runs sparse recurrent networks wired from MaleCNS v1.0. A typing circuit drives foreleg motor neurons; a separate thinker circuit learns code generation and edits.
- **`training/`** provides imitation and PPO, teacher-generated data, distillation, evaluation, and a resumable nine-stage curriculum: keys → transcription → editing → recovery → oracle data → distillation → micro-programs → longer programs → self-correction.
- **`web/`** is the Three.js replay viewer, served by the Python backend. It shows scripted kinematic, scripted physics, or learned-policy replays, plus modelled neural activity.

All nine curriculum stages have passed smoke checks with relaxed gates. Full curriculum training and a connectome advantage remain unproven. Motor observations include the next key's location; the thorax is carried kinematically with gravity off, and Fn chords use a sticky latch. See [training details](docs/training.md), [research results and limits](docs/results.md), [architecture contracts](docs/architecture.md), and [asset provenance](THIRD_PARTY.md).

## Run the viewer

Install uv and Python 3.11 or 3.12, then run from the repository root:

```sh
uv sync
uv run python -m backend.server
```

Open <http://127.0.0.1:8000>, select a target and replay source, and use the timeline to inspect typing. Replays are generated on first request and cached.

After editing the browser source, rebuild its bundled assets (Node.js 22+):

```sh
cd web
npm ci
npm run build
```

## Run training

The full curriculum uses an MLX teacher on Apple silicon. Training respects a configurable RAM cap, default **4 GB**; `--ram-gb` overrides `FLYSLOP_MAX_RAM_GB`, which overrides the config.

```sh
uv sync --extra training --extra teacher

# Check all nine stages with small budgets and relaxed gates.
uv run --extra training --extra teacher python -u -m training.curriculum \
    --config training/curriculum.json --smoke --advance-on-budget --ram-gb 2

# Full training; run offline, as stages can take hours or days.
uv run --extra training --extra teacher python -u -m training.curriculum \
    --config training/curriculum.json --ram-gb 4 --wandb-project

# Resume an interrupted run using its printed directory.
uv run --extra training --extra teacher python -u -m training.curriculum \
    --resume runs/curriculum/<run> --ram-gb 4
```

The curriculum stops when a gate fails or a budget expires. Add `--advance-on-budget` to continue past exhausted budgets. Run directories contain `state.json`, stage manifests, and checkpoints.

Training progress is printed live and saved to `runs/curriculum/<run>/training.jsonl`: stage starts/results, BC loss, PPO progress, evaluations, and error tracebacks. Follow it from another terminal:

```sh
tail -f runs/curriculum/<run>/training.jsonl
```

Thinker training also writes per-step loss, accuracy, and timing to `schedule.jsonl` under its output directory; rows with an `event` key describe schedule transitions. Stage 5 writes teacher output to `stage5/oracle_gen.log`. Standalone PPO writes `eval.jsonl` and TensorBoard curves:

```sh
uv run --extra training tensorboard --logdir runs
```

For optional Weights & Biases tracking, install the `tracking` extra and enable a project:

```sh
uv sync --extra training --extra teacher --extra tracking
uv run --extra tracking wandb login
uv run --extra training --extra teacher --extra tracking python -u -m training.curriculum \
    --config training/curriculum.json --ram-gb 4 --wandb-project flyslop
```

W&B records curriculum configuration, BC/PPO loss and performance, thinker loss/accuracy, evaluations, and stage status. Use `--wandb-mode offline` to record locally without logging in, or `--wandb-entity TEAM` to select a workspace. `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_MODE` also work. Tracking is disabled without a project; local logs remain available. Add the same project/entity flags to `--resume` to continue an online W&B run; offline resumes create separate sessions.

For the shorter imitation → PPO recipe and evaluation commands, see [training.md](docs/training.md). To view a trained physical policy:

```sh
uv run --extra training python -m backend.server --policy runs/bc/seed0/bc_model.zip
```

Choose **Learned policy · physics** in the viewer. `--policy` accepts BC or PPO `.zip` checkpoints.

## Verify

```sh
uv run python scripts/validate_corpus.py
uv run --extra training python -m unittest discover -s tests
```
