# Physics keyboard (package C)

`backend/physics.py` builds a MuJoCo scene from the vendored NeuroMechFly v2
tree (`data/neuromechfly/model.json`), a life-size fly on a fly-sized laptop
(0.3 mm key pitch; see [coordinates.md](coordinates.md)), in key-pitch units. No
FlyGym runtime is needed.

- **Legs:** 6 × 7 hinge joints (Coxa yaw/pitch/roll, Femur, Femur roll,
  Tibia, Tarsus1), each driven by a position actuator (`kp=800`, `kv=16`, times `joint_gain`). The
  fixed tarsal joints are baked into the body frames. MuJoCo claw positions
  match `backend.flybody` forward kinematics to 1e-6 key pitches (unit test).
  Gains were tuned for a fly 10/19.05 key pitches per mm. Leg inertia grows as
  scale³, so `joint_gain(scale)` multiplies kp, kv, armature, and joint damping
  by `(scale / TUNED_SCALE)**3`. This keeps the tuned leg dynamics at any fly-to-keyboard size.
- **Contacts:** only claw spheres (radius 0.015) collide, with keycaps and
  the deck. The deck is 93 boxes that leave holes under every key, so a
  pressed key can travel below the deck surface as on a laptop. A palm-rest
  box in front of the keyboard carries the hind legs. Deck boxes are 1 unit
  thick, so claws cannot tunnel through.
- **Keys:** every key is a slide joint with stiffness 300, preload 0.1, and
  damping 30. All keys have the same 4-unit mass, as long keys have
  stabilizers. Travel is 0.075.
- **Typing rule (`PhysicalKeyboard`):** a character is emitted when a key
  passes `press_depth = 0.05` *and* a claw touched that cap within 10 ms. The
  key re-arms below `release_depth = 0.02`. Events use the same schema as
  `ContactKeyboard`, so `replay_text` validates physics replays.
- **Stepping:** the tick rate is 30 Hz with a 1 ms timestep (33 substeps).
  Joint targets ramp linearly across a tick instead of stepping.

## Limitations

- The thorax is carried by a mocap body. Legs do not support weight and the
  body does not react to leg forces. Gravity is off.
- Key force is a game-scale mechanic, not a fruit-fly force.
- Walking is a scripted tripod gait. The scripted planner provides the gate
  demonstration. Inside `FlyTypingEnv`, both forelegs are policy controlled
  and the four middle and hind legs walk in diagonal pairs.

## Gate result (PLAN §5.3, scripted body)

Command: `uv run python scripts/evaluate_physics_gate.py --output docs/physics_gate.json`.
It types seeded random strings over the corpus alphabet, about 200 physical
presses per seed, 3 seeds. See [physics_gate.json](physics_gate.json) and
[results.md](results.md) for the numbers. "Uncredited actuations" counts keys
kicked past the threshold by a landing foot without a claw on the cap. They
emit no characters but are reported, because a real keyboard might register
them.

## Tuning notes

Found during bring-up, in case the constants are changed:

- MuJoCo's default soft contacts (`solref` 0.02 s) let claws sink through
  caps. Contacts now use a 4 ms time constant.
- A solid deck under the keys stopped claws at −0.04, so it now has holes.
- Light, underdamped keys were flicked past the threshold after the claw had
  left. Heavier damped keys, a preload, and the touch window fixed this.
- Density-based mass made the space bar about 6× heavier than letters, so it
  could not be pressed. Mass is now uniform.
- Stepping joint targets once per tick slammed feet into keys. Targets now
  ramp across substeps.
- The fly-sized laptop (the fly got about 6× bigger relative to the keys)
  needed: kv 8 → 16, because walking legs landed hard enough to type; a body
  lean during presses scaled by `1/MM`, because the old 0.09 rad pitch pushed
  the hind feet through the deck; slower (0.3 u/tick), higher, settled
  foreleg reaches; a 5-tick gait swing with a sin² soft landing; and a 0.55 u
  inset on long keys, because claw lag of about 0.3 u hit the neighbouring key.
- In `FlyTypingEnv`, the foreleg target is rate limited in the body frame.
  In the world frame it lagged the walking body, left the reach box, and the
  leg dropped onto keys.
