"""Laptop (US ANSI, MacBook-style) keyboard geometry and a contact-gated switch.

Layout coordinates: one key pitch is 1.0; ``x`` grows to the right,
``y`` grows toward the user, each key's ``x,y`` is its top-left corner, and
the keycap top surface is z=0. Every row spans 15 units. A foot is a point
contact. This is a game-scale proxy, not a claim that an actual fruit fly can
actuate a laptop switch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

REAL_KEY_PITCH_MM = 19.05
# The laptop is shrunk to fly size: a 0.3 mm key pitch (about 1/64 scale)
# against a life-size fly, so the fly stands over several keys and reaches most
# of them with its forelegs. Only the length unit changes; layout numbers stay in pitches.
KEY_PITCH_MM = 0.3
LAPTOP_SCALE = KEY_PITCH_MM / REAL_KEY_PITCH_MM
KEY_GAP = 0.1          # space between neighbouring keycaps, in key pitches
KEY_TRAVEL = 0.075     # bottom-out depth, in key pitches
DECK_DEPTH = 0.04      # keyboard deck below the keycap tops, in key pitches


@dataclass(frozen=True)
class Key:
    id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    output: str
    shift_output: str


def build_layout() -> list[Key]:
    keys: list[Key] = []

    def row(y: float, entries: list[tuple], height: float = 1.0) -> None:
        """Place a physical row from left to right; widths are key pitches."""
        x = 0.0
        for key_id, label, output, shifted, width in entries:
            keys.append(Key(key_id, label, round(x + KEY_GAP / 2, 4), round(y + KEY_GAP / 2, 4),
                            round(width - KEY_GAP, 4), round(height - KEY_GAP, 4), output, shifted))
            x += width

    def printable(plain: str, shifted: str) -> list[tuple]:
        return [(a, a.upper() if a.isalpha() else a, a, b, 1.0) for a, b in zip(plain, shifted)]

    row(0.0, [("Escape", "esc", "", "", 1.5)] + [(f"F{i}", f"F{i}", "", "", 1.0) for i in range(1, 13)]
        + [("Power", "⏻", "", "", 1.5)], height=0.6)
    row(0.6, printable("`1234567890-=", "~!@#$%^&*()_+") + [("Backspace", "delete", "", "", 2.0)])
    row(1.6, [("Tab", "tab", "", "", 1.5)] + printable("qwertyuiop[]", "QWERTYUIOP{}") + [("\\", "\\", "\\", "|", 1.5)])
    row(2.6, [("CapsLock", "caps lock", "", "", 1.75)] + printable("asdfghjkl;'", 'ASDFGHJKL:"')
        + [("Enter", "return", "\n", "\n", 2.25)])
    row(3.6, [("ShiftLeft", "shift", "", "", 2.25)] + printable("zxcvbnm,./", "ZXCVBNM<>?")
        + [("ShiftRight", "shift", "", "", 2.75)])
    row(4.6, [("Fn", "fn", "", "", 1.0), ("ControlLeft", "control", "", "", 1.0), ("AltLeft", "option", "", "", 1.0),
              ("MetaLeft", "command", "", "", 1.25), ("Space", "", " ", " ", 5.5), ("MetaRight", "command", "", "", 1.25),
              ("AltRight", "option", "", "", 1.0), ("ArrowLeft", "◀", "", "", 1.0)])
    # Inverted-T arrows: half-height up/down share one column.
    keys.append(Key("ArrowUp", "▲", 13.05, 4.65, 0.9, 0.4, "", ""))
    keys.append(Key("ArrowDown", "▼", 13.05, 5.15, 0.9, 0.4, "", ""))
    keys.append(Key("ArrowRight", "▶", 14.05, 5.15, 0.9, 0.4, "", ""))
    return keys


LAYOUT = build_layout()
BY_ID = {key.id: key for key in LAYOUT}
LAYOUT_WIDTH = 15.0
LAYOUT_HEIGHT = 5.6
CHAR_TO_KEY: dict[str, tuple[str, bool]] = {}
for _key in LAYOUT:
    if _key.output:
        CHAR_TO_KEY[_key.output] = (_key.id, False)
    if _key.shift_output and _key.shift_output != _key.output:
        CHAR_TO_KEY[_key.shift_output] = (_key.id, True)


SHIFT_KEYS = ("ShiftLeft", "ShiftRight")
MODIFIER_KEYS = ("ShiftLeft", "ShiftRight", "Fn")   # the only modifiers with meaning (contract C2)
TAB_TEXT = " " * 4
EDIT_KEYS = ("Backspace", "Tab", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown")


def key_char(key_id: str, mods=()) -> str:
    """Text a key inserts given the held modifiers; "" for Backspace, arrows and dead keys."""
    if key_id == "Tab":
        return TAB_TEXT
    key = BY_ID[key_id]
    return key.shift_output if any(m in SHIFT_KEYS for m in mods) else key.output


def emits_key(key_id: str, character: str) -> bool:
    """Whether a press yields a ``key`` event: inserting keys plus the editing keys."""
    return bool(character) or key_id in EDIT_KEYS


def layout_payload() -> dict:
    return {"keys": [asdict(key) for key in LAYOUT], "unit": "one key pitch", "key_pitch_mm": KEY_PITCH_MM,
            "laptop_scale": LAPTOP_SCALE, "width": LAYOUT_WIDTH, "height": LAYOUT_HEIGHT, "surface_z": 0.0,
            "travel": KEY_TRAVEL, "deck_depth": DECK_DEPTH}


def hit_test(x: float, y: float) -> str | None:
    for key in LAYOUT:
        if key.x <= x <= key.x + key.width and key.y <= y <= key.y + key.height:
            return key.id
    return None


def surface_z(x: float, y: float) -> float:
    """Resting height for a foot: keycap top, or the deck between keys."""
    return 0.0 if hit_test(x, y) is not None else -DECK_DEPTH


class ContactKeyboard:
    """One output per completed down/up cycle per foot; no output from hovering.

    ``shift_mode="held"``: Shift and Fn are active while their key contact persists (or
    until :meth:`release`); a chord is the modifier's contact still active when the main
    key's contact begins. ``shift_mode="latch"`` is the legacy one-shot Shift: a Shift
    press shifts the next emitted key. Every contact gets an integer ``contact_id``.
    """

    def __init__(self, debounce_ticks: int = 2, press_depth: float = 0.05, release_height: float = 0.25,
                 shift_mode: str = "latch"):
        if shift_mode not in ("held", "latch"):
            raise ValueError("shift_mode must be 'held' or 'latch'")
        self.debounce_ticks = debounce_ticks
        self.press_depth = press_depth
        self.release_height = release_height
        self.shift_mode = shift_mode
        self.down: dict[str, str] = {}
        self.armed: dict[str, bool] = {}
        self.contact_ids: dict[str, int] = {}
        self.released: set[str] = set()   # modifiers released by command while still in contact
        self.next_contact_id = 0
        self.shift_latched = False
        self.last_emission_tick = -10_000

    def hit_test(self, x: float, y: float) -> str | None:
        return hit_test(x, y)

    def held_modifiers(self) -> list[str]:
        """Modifier ids whose contact is active now, in stable order."""
        active = set(self.down.values()) - self.released
        return [m for m in MODIFIER_KEYS if m in active]

    def release(self, tick: int, key_id: str) -> list[dict]:
        """Explicit release command for a held modifier whose foot is still down."""
        if key_id not in MODIFIER_KEYS or key_id not in self.held_modifiers():
            return []
        self.released.add(key_id)
        return [{"tick": tick, "type": "modifier", "key_id": key_id, "state": modifier_state(key_id, False)}]

    def update(self, tick: int, x: float, y: float, z: float, shifted: bool = False,
               foot: str = "right_foreleg") -> list[dict]:
        events: list[dict] = []
        candidate = hit_test(x, y)
        down_key = self.down.get(foot)
        if down_key is not None and (z >= self.release_height or candidate != down_key):
            was_held = down_key in self.held_modifiers()
            events.append({"tick": tick, "type": "contact_offset", "key_id": down_key, "foot": foot,
                           "contact_id": self.contact_ids.pop(foot)})
            del self.down[foot]
            if down_key not in self.down.values():
                self.released.discard(down_key)
            if was_held and down_key not in self.down.values() and (self.shift_mode == "held" or down_key == "Fn"):
                events.append({"tick": tick, "type": "modifier", "key_id": down_key, "foot": foot,
                               "state": modifier_state(down_key, False)})
            down_key = None
        if z >= self.release_height:
            self.armed[foot] = True
        if self.armed.get(foot, True) and down_key is None and z <= -self.press_depth and candidate is not None:
            self.down[foot] = candidate
            self.armed[foot] = False
            contact_id = self.next_contact_id
            self.next_contact_id += 1
            self.contact_ids[foot] = contact_id
            events.append({"tick": tick, "type": "contact_onset", "key_id": candidate, "foot": foot,
                           "contact_id": contact_id})
            held_mode = self.shift_mode == "held"
            if candidate in MODIFIER_KEYS and (held_mode or candidate == "Fn"):
                events.append({"tick": tick, "type": "modifier", "key_id": candidate, "foot": foot,
                               "state": modifier_state(candidate, True)})
            if tick - self.last_emission_tick >= self.debounce_ticks:
                if not held_mode and candidate in SHIFT_KEYS:
                    self.shift_latched = True
                    events.append({"tick": tick, "type": "modifier", "key_id": candidate, "foot": foot, "state": "shift_latched"})
                mods = [m for m in self.held_modifiers() if m != candidate]
                latched = not held_mode and self.shift_latched
                if held_mode:
                    if shifted and not any(m in SHIFT_KEYS for m in mods):
                        mods.insert(0, "ShiftLeft")
                    character = key_char(candidate, mods)
                else:
                    character = key_char(candidate, ["ShiftLeft"] if (latched or shifted) else [])
                    if latched or shifted:
                        mods = [m for m in mods if m not in SHIFT_KEYS] + ["ShiftLeft"]
                if emits_key(candidate, character):
                    event = {"tick": tick, "type": "key", "key_id": candidate, "key": candidate, "foot": foot,
                             "modifiers": mods, "char": character, "contact_id": contact_id}
                    if latched:
                        event["shift_latched"] = True
                    events.append(event)
                    self.last_emission_tick = tick
                    self.shift_latched = False
        return events


def modifier_state(key_id: str, down: bool) -> str:
    return f"{'fn' if key_id == 'Fn' else 'shift'}_{'down' if down else 'up'}"
