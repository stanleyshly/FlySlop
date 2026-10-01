# Coordinate frames and units

## Keyboard layout frame

**One key pitch equals 1.0 unit.** The laptop is shrunk to fly size: `KEY_PITCH_MM = 0.3` (about 1/64 of a real 19.05 mm pitch, `LAPTOP_SCALE`). `backend/keyboard.py` defines a US ANSI laptop layout in the style of a MacBook: a half-height function row, number row, QWERTY rows, and a bottom row with fn/control/option/command, a 5.5u space bar, and inverted-T arrows. Every row spans 15 units. Each key's `x,y` is the top-left corner of its cap, `x` grows to the right and `y` grows toward the user. Caps are 0.1u smaller than their pitch, which leaves the gaps. `width,height` are cap dimensions, and the keycap top surface is `z = 0`. The deck between keys is `z = -0.04` (`DECK_DEPTH`), and a key bottoms out at `z = -0.075` (`KEY_TRAVEL`).

## World frame

`backend/embodiment.py` uses a right-handed world frame in key pitches:

| Axis | Meaning | Positive direction |
| --- | --- | --- |
| `X` | Across the keyboard | Right |
| `Y` | Front to back | Away from the user, toward the screen |
| `Z` | Height | Up; `Z = 0` is the keycap top |

The origin is the centre of the keyboard layout: `X = x - 7.5`, `Y = 2.8 - y`. The viewer's `simRoot` maps world `(X, Y, Z)` to three.js `(X, Z, -Y)`. The laptop base, trackpad, and hinged display are drawn around this origin, so the keyboard is centred on the laptop.

## Fly body frame and scale

The NeuroMechFly v2 kinematic tree (`data/neuromechfly/model.json`) is in NeuroMechFly millimetres. The thorax frame has `x` forward (head), `y` left, and `z` up. The fly is life size (`FLY_SCALE = 1`) and the laptop is scaled down instead, so one fly millimetre is `1 / 0.3 ≈ 3.3` key pitches. The body is about 11 key pitches long and stands over the keys. Each foreleg reaches an ellipse around its press anchor (thorax frame `(0.85, ±0.75)` mm, half-axes 0.6 mm forward and 0.9 mm lateral), about 4 × 6 keys, so most keys are pressed without stepping. The hind legs stand on the palm rest. This is a game scale, not a claim about real fly or laptop mechanics. Replay frames record the thorax pose as `x, y, z, yaw, pitch, roll`. The rotation is `Rz(yaw) · Ry(pitch) · Rx(roll)`, so `yaw = π/2` means the fly faces the screen.

## Joints and contact geometry

Each replay frame stores seven angles per leg in `model.leg_dofs` order (`Coxa_yaw, Coxa, Coxa_roll, Femur, Femur_roll, Tibia, Tarsus1`). Joints outside that list, including the distal tarsal segments, stay at the rest pose recorded in `model.json`. Hinge rotations compose as in MuJoCo, and the export script verifies this against `mujoco.mj_kinematics`. Each frame also stores the world-space claw tip of every leg. The tip is the Tarsus5 mesh vertex farthest from its joint, after IK.

Contact uses those posed tips. All six legs pass through `ContactKeyboard`, each with its own down/armed state. A press begins when a tip is over a key rectangle and at `z <= -0.05`. Contact ends when that tip rises to `z >= 0.25` or leaves the key, and a new press on the same foot is armed only after that rise. Stance feet rest at `z = 0` on caps or `-0.04` on the deck, so they cannot type. The scripted strike takes the foreleg to `-0.075`. The viewer depresses a keycap by the depth of any tip below its surface, and it highlights the cap from contact events. It never infers contact itself.

These thresholds are game mechanics. The model does not represent switch force, compliance, friction, leg mass, or a fly's ability to actuate a laptop key.

## Time

Replay time is an integer tick at `metadata.tick_hz = 30`, and `frames.rows[tick]` holds that tick's pose. The tripod gait swings each group for 3 ticks (5 Hz stepping). The frontend interpolates between frames for drawing, but event ticks remain authoritative.

## Comparison with FlyLab

The audited FlyLab source uses a different scene frame and apparent metre-scale values (for example, keyboard centers near `y=1.385`, key widths around `0.105`, `DT=0.002` seconds, hence 500 simulation ticks per second). Its frontend applies local transforms to seat the body and its own keyboard geometry. Those numbers must not be copied into FlySlop as if they shared units. Translate geometry only through an explicit scale and frame transform, then keep the emitted FlySlop layout as the contact authority.
