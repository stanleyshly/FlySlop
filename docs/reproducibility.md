# Reproducing the current prototype

Run all commands from the repository root unless a step says otherwise. Python is constrained to `>=3.11,<3.13` in `pyproject.toml`; `uv.lock` pins the Python dependency resolution. `web/package-lock.json` pins the browser dependency tree. The authored corpus and replay carry SHA-256 digests, and the three copied student snippets retain their source commit and byte hashes in `data/corpus/manifest.jsonl`.

```sh
uv sync --frozen
uv run python scripts/validate_corpus.py
uv run python scripts/evaluate_corpus_oracle.py
uv run python -m unittest discover -s tests -v
uv run python -m backend.demo --target fly_demo --output data/replays/fly_demo.json
uv run python -m backend.server
```

Open <http://127.0.0.1:8000>. The server reads a target from the manifest, generates a deterministic kinematic episode, verifies replay integrity, then sends the event log to the client. The client never generates scored key events. A replay is a JSON artifact and can be inspected without running the controller.

To rebuild browser assets, run `npm ci` and `npm run build` in `web/`; the build writes static files in `web/` for the Python server. `web/node_modules/` is local and excluded from version control.

## Determinism and scoring

The current scripted episode runs at 30 ticks per second. An open-loop planner walks the NeuroMechFly thorax with a tripod gait and turns toward long walks. It then lifts a foreleg and presses: reach, touch, push to key travel, hold, and release. Batched inverse kinematics on the exported NeuroMechFly tree poses all six legs. A key event is emitted only when a posed claw tip is inside a key rectangle and below the travel threshold. `replay_text` checks monotonic ticks, matching onset/offset pairs, active contact for every key event, exact buffer snapshots, and SHA-256 buffer hashes. The judge reports exact transcription, syntax, and elaboration separately. No functional testbench result is currently reported.

The sample target is an original authored standalone SV module. The client displays the visible target, which is the specified transcription condition. The script reads target characters to choose keys, so its exact score is an **oracle baseline**, not a learning result. `metadata.mode` and `observation_condition` record this status.

## Limits to reproduce research claims

The keyboard uses one ordinary key pitch as a simulator unit; its travel threshold is a scaled game mechanic. The body is the NeuroMechFly v2 kinematic tree at life size, on a laptop scaled to 0.3 mm keys. It is posed kinematically, and each claw tip is a point contact. No measured force, gait stability, walking physics, or realistic fruit fly actuation is present. The visual activity sample comes from leg height, stance, and press state. It has no connectome wiring or measured neural dynamics. The kinematic replay alone establishes only the contact simulation claim. The MuJoCo body, learned policies, and connectome actor are described in [physics.md](physics.md), [training.md](training.md), and [connectome.md](connectome.md). Their measured status is in [results.md](results.md).

Physics and learning:

```sh
uv sync --extra training --extra connectome
uv run python scripts/evaluate_physics_gate.py --output docs/physics_gate.json   # ~40 s
uv run python -m backend.demo --target fly_demo --source physics --output data/replays/fly_demo.physics.json
uv run --extra training --extra connectome python -m unittest discover -s tests -v
```

Training commands, the smoke tests, and the full five-seed sweeps are in [training.md](training.md). The physics keyboard is in [physics.md](physics.md), and the connectome data and ablation in [connectome.md](connectome.md). Physics episodes are deterministic for a given seed and MuJoCo version.

See [docs/results.md](results.md) for the measured checks and [docs/reuse-audit.md](reuse-audit.md) for the pinned FlyLab assessment.
