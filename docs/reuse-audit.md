# FlyLab reuse audit

Audit date: 2026-09-26. Source: [Sudharsanselvaraj/Flylab](https://github.com/Sudharsanselvaraj/Flylab), cloned to `/tmp/flylab-audit` at commit `511b12c8c5b2c3a5a6efb22426e5fb30a256243e` (`main` at audit time). The checkout was kept outside this repository and is not a dependency.

## Decision

**Reuse the interaction and presentation design selectively; do not adopt FlyLab as this project's simulation backend or training system.** Its MIT license permits adaptation with the required copyright and license notice. FlyLab demonstrates the key causal boundary that this project needs: backend-controlled movement, geometric contact, then native key input and an inspectable event timeline. Its React/Three.js workroom is a useful UI reference and a potential source for selected components after component-level dependency and attribution review.

FlyLab does not supply the physical embodiment or task needed here. Its body is seated; the foreleg effectors move kinematically while the body does not walk or translate. Its published task is a fixed-layout, single-line HTML heading benchmark. The current FlySlop prototype therefore uses its own smaller, deterministic contact and replay contract. FlyGym remains a candidate if a later milestone needs articulated walking physics; it is not needed for the first contact-to-text demonstration.

Do not import FlyLab's trained controller, checkpoints, cached connectome, or reported score as evidence for FlySlop. The policy and modeled neural dynamics are tied to FlyLab's pixel benchmark and are explicitly described upstream as engineered rather than dynamics-validated. Any future reuse of those assets needs its own provenance and experiment review.

## Findings

| Question | Finding at pinned commit |
| --- | --- |
| License | MIT, copyright S Sudharsan (2026), in `LICENSE`. Preserve the license and copyright notice for any copied or adapted substantial code. |
| Stack | FastAPI and Python 3.11+; React 19, Three.js, Zustand, and Node.js 22+. `services/sim-engine/pyproject.toml` pins Playwright to 1.58.0 but its other Python dependencies are mostly ranges. `web/app/package-lock.json` locks the client dependency tree. |
| Fly movement | The body is a fixed seated model. Two front-leg endpoints (`effector`, `left_effector`) are updated by scripted interpolated poses. Other four legs are static. There is no body translation, walking gait, articulated joint dynamics, or force model. |
| Contact | Geometric and kinematic. `ComputerActuator.contact()` checks endpoint position against the key rectangles and a vertical band. Chromium keyboard input is withheld if there is no key contact. This is a simulator gate, not rigid-body collision physics. |
| Monitor and multiline text | A real Chromium workspace screenshot is shown. The editor is a `<textarea>` with `white-space: pre` and native keyboard input, so newline characters can be entered. The documented benchmark is still one-line HTML; the fixed 400×420 editor and 400×420 preview are not evidence of a tested multiline SystemVerilog workflow. |
| Neural influence | Yes, in the published experiment: screenshots feed a trained visual readout/edit policy, which advances the connectome-inspired rate state and selects the next action. The anatomy/connectivity is based on cached MaleCNS data; dynamics are an engineered rate model. This does not establish biological causality. |
| Training/evaluation | Training scripts and saved checkpoints/evidence are separate from live sessions. The privileged evaluator reads source and scores screenshots after actions; upstream says target source, DOM state, and reward are not passed to the policy. The headline result is bounded to its fixed-layout heading task. |
| Replay | The session uses an integer neural tick and archives decisions, screenshots, event timeline, activity, and hashes. The browser/client render and record events; they do not synthesize keystrokes. |

## Commands attempted

Commands below ran from the isolated checkout unless noted.

| Command | Result |
| --- | --- |
| `git ls-remote https://github.com/Sudharsanselvaraj/Flylab.git HEAD refs/heads/main` | Both refs resolved to `511b12c8c5b2c3a5a6efb22426e5fb30a256243e`. |
| `git clone --depth 1 https://github.com/Sudharsanselvaraj/Flylab.git /tmp/flylab-audit` | Succeeded; `git -C /tmp/flylab-audit rev-parse HEAD` returned the same hash; worktree was clean. |
| `PYTHONPATH=services/sim-engine python3 -m pytest services/sim-engine/tests -q` | Could not start: system Python 3 reported `No module named pytest`. The upstream Python environment and optional anatomy/connectome caches were not installed. |
| `npm ci` (in `web/app`) | Succeeded; installed 223 packages, audit reported 0 vulnerabilities. npm emitted its `fsevents` install-script approval warning. |
| `npm run build` (in `web/app`) | Succeeded: TypeScript and Vite production build completed. Vite warned that the minified JavaScript chunk is 1,162.01 kB, above its 500 kB advisory threshold. |
| `npm run demo -- --host 127.0.0.1` (in `web/app`), then `curl` against the preview | Build and static preview startup succeeded; `GET /` returned 200. The client API request `GET /api/coding/catalog` returned 502 because the FastAPI service was not running on `127.0.0.1:8050`. The preview was stopped after this check. |

The full documented live run was not attempted: it needs the Python simulation environment, Playwright Chromium, and prepared FlyLab data caches. A successful frontend build checks compilation only; it does not verify the Python/browser session, trained policy, or complete demo. No runtime or throughput number is inferred from the README's reported experiments.

## Source references

- [Pinned source README](https://github.com/Sudharsanselvaraj/Flylab/blob/511b12c8c5b2c3a5a6efb22426e5fb30a256243e/README.md)
- [Pinned MIT license](https://github.com/Sudharsanselvaraj/Flylab/blob/511b12c8c5b2c3a5a6efb22426e5fb30a256243e/LICENSE)
- Relevant inspected files: `services/sim-engine/hawking_fly/physical_coding/motor.py`, `browser.py`, `session.py`, `policy.py`; `web/app/src/features/coding/PhysicalWorld.tsx`, `DeskSeat.tsx`; `docs/architecture.md`.
