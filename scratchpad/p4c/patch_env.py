import re
p='backend/fly_env.py'
s=open(p).read()
def rep(old,new,count=1):
    global s
    assert s.count(old)==count,(old,s.count(old))
    s=s.replace(old,new)

rep('''RF, LF = LEGS.index("RF"), LEGS.index("LF")''','''SHIFT_IDS = ("ShiftLeft", "ShiftRight")
CHORD_REACH = 0.8        # max reach-ellipse ratio (1 = edge of a foreleg's reach box) for a chord's two press points
CHORD_GRID = 0.25        # thorax XY search resolution for chord standing spots (key pitches)

RF, LF = LEGS.index("RF"), LEGS.index("LF")''')

rep('''def obs_labels(''','''class UnreachableChord(ValueError):
    """No thorax pose lets the two forelegs hold the modifier and press the key at the same time."""


def normalize_key_items(keys) -> list[tuple[str, bool, tuple[str, ...]]]:
    """Key-command queue items -> ``(key_id, shifted, extra)``.

    Accepted: legacy ``(key_id, shifted)`` tuples, ``(key_id, mods)`` with ``mods`` a sequence of modifier ids, and
    C2 commands (dicts or objects with ``key``, ``mods``, ``hold``, ``release``). ``hold``/``release`` of a Shift key
    are folded into the items between them. ``extra`` holds ``"Fn"`` and/or an explicit Shift key id.
    """
    out, sticky = [], None
    for item in keys:
        hold = release = False
        shifted = False
        if isinstance(item, dict) or hasattr(item, "mods"):
            get = item.get if isinstance(item, dict) else (lambda name, default=None: getattr(item, name, default))
            key, mods = str(get("key")), tuple(get("mods", ()) or ())
            hold, release = bool(get("hold", False)), bool(get("release", False))
        else:
            item = tuple(item)
            key, second = str(item[0]), (item[1] if len(item) > 1 else False)
            if isinstance(second, (bool, int, np.bool_)):
                shifted, mods = bool(second), (tuple(item[2]) if len(item) > 2 else ())
            else:
                mods = tuple(second)
        if key not in BY_ID:
            raise ValueError("unknown key id in queue")
        bad = [m for m in mods if m not in SHIFT_IDS and m != "Fn"]
        if bad:
            raise ValueError(f"modifiers {bad} are not physically used (only Shift and Fn, contract C2)")
        if release:
            sticky = None
            continue
        if hold:
            if key not in SHIFT_IDS:
                raise ValueError("only a Shift key can be held")
            sticky = key
            continue
        shift_id = next((m for m in mods if m in SHIFT_IDS), sticky)
        extra = (("Fn",) if "Fn" in mods else ()) + ((shift_id,) if shift_id else ())
        out.append((key, shifted or shift_id is not None, extra))
    return out


def obs_labels(''')

rep('''        self.queue: list | None = None       # key-command queue mode: [(key_id, shifted), ...] (see reset options["keys"])
        self.qi = 0''','''        self.queue: list | None = None       # key-command queue mode: [(key_id, shifted, extra), ...] (see reset options["keys"])
        self.qi = 0
        self._chord_cache: dict = {}
        self._chord_plan_cache = None
        self._reach_planner: Planner | None = None''')

# typing_leg
rep('''        key_id = self.next_key()
        if key_id is None:
            return "RF"
        point = press_point(key_id, float(self.body[0]))''','''        key_id = self.next_key()
        if key_id is None:
            return "RF"
        chord = self._current_chord()
        if chord is not None:
            return self._chord_plan(chord)["key_leg"]
        point = press_point(key_id, float(self.body[0]))''')

rep('''        keys = [(str(k), bool(sh)) for k, sh in keys]
        if any(k not in BY_ID for k, _ in keys):
            raise ValueError("unknown key id in queue")
        self.queue, self.qi = keys, int(start)
        self.target, self.buffer = "\\0" * len(keys), "\\0" * self.qi
''','''        keys = normalize_key_items(keys)
        if not keys:
            raise ValueError("keys must be non-empty")
        for item in keys:
            self.check_chord(item)
        self.queue, self.qi = keys, int(start)
        self.target, self.buffer = "\\0" * len(keys), "\\0" * self.qi
        self._chord_plan_cache = None

    # -- held-modifier chords (contract C2) -------------------------------
    def _current_item(self):
        """``(key_id, shifted, extra)`` of the press the env is waiting for, in queue or text mode."""
        if self.queue is not None:
            return self.queue[self.qi] if self.qi < len(self.queue) else None
        if len(self.buffer) >= len(self.target):
            return None
        key_id, shifted = CHAR_TO_KEY[self.target[len(self.buffer)]]
        return key_id, shifted, ()

    def _hold_mods(self, item) -> list[str]:
        """Modifiers a foreleg must hold down while ``item``'s key is struck: ``"Fn"``, ``"Shift"`` (either Shift key)
        or an explicit Shift id. Shift only counts in ``shift_mode="held"``, Fn only in ``fn_mode="held"``."""
        p = self.sim.params
        mods = []
        if "Fn" in item[2] and p.fn_mode == "held":
            mods.append("Fn")
        if item[1] and p.shift_mode == "held":
            mods.append(next((m for m in item[2] if m in SHIFT_IDS), "Shift"))
        return mods

    def _fn_tap_pending(self, item) -> bool:
        return "Fn" in item[2] and self.sim.params.fn_mode == "latch" and not self.sim.keyboard.fn_latched

    def _current_chord(self):
        """The current item when it must be pressed as a two-foreleg chord, else None (single-leg paths)."""
        if self.queue is None and self.sim.params.shift_mode != "held":
            return None
        item = self._current_item()
        if item is None or not self._hold_mods(item) or self._fn_tap_pending(item):
            return None
        return item

    def _planner(self) -> Planner:
        if self._reach_planner is None:
            self._reach_planner = Planner()
        return self._reach_planner

    def _pair_ratio(self, key_id: str, mod_id: str, mod_leg: str, key_leg: str, xy) -> float:
        planner, body = self._planner(), self._planner().body_at(np.asarray(xy, dtype=float))
        worst = 0.0
        for leg, target in ((mod_leg, mod_id), (key_leg, key_id)):
            anchor = planner.body_point(body, PRESS_ANCHOR[leg])
            point = press_point(target, float(anchor[0]))
            worst = max(worst, planner.reach_ratio(body, leg, point))
        return worst

    def chord_options(self, key_id: str, mod: str) -> list:
        """Every ``(ratio, modifier id, modifier leg, key leg, thorax xy)`` from which both forelegs reach their keys
        (ratio <= ``CHORD_REACH``). ``mod`` is ``"Fn"``, ``"Shift"`` (either key) or a Shift key id. Cached."""
        cache_key = (key_id, mod)
        if cache_key not in self._chord_cache:
            ids = SHIFT_IDS if mod == "Shift" else (mod,)
            (x0, x1), (y0, y1) = BOUNDS
            options = []
            for mod_id in ids:
                for mod_leg, key_leg in (("LF", "RF"), ("RF", "LF")):
                    for x in np.arange(x0, x1 + 1e-9, CHORD_GRID):
                        for y in np.arange(y0, y1 + 1e-9, CHORD_GRID):
                            ratio = self._pair_ratio(key_id, mod_id, mod_leg, key_leg, (x, y))
                            if ratio <= CHORD_REACH:
                                options.append((ratio, mod_id, mod_leg, key_leg, np.array([x, y])))
            self._chord_cache[cache_key] = options
        return self._chord_cache[cache_key]

    def chord_span(self, key_id: str, mod: str) -> float:
        """Horizontal distance (key pitches) between the modifier's and the key's nearest press points."""
        best = min((abs(press_point(m, float(press_point(key_id, 0.0)[0]))[0] - press_point(key_id, 0.0)[0])
                    for m in (SHIFT_IDS if mod == "Shift" else (mod,))), default=0.0)
        return float(best)

    def check_chord(self, item) -> None:
        """Raise :class:`UnreachableChord` when ``item`` cannot be realised with two held forelegs."""
        mods = self._hold_mods(item)
        if len(mods) > 1:
            raise UnreachableChord(f"{'+'.join(mods)}+{item[0]}: three keys down at once, only two forelegs press")
        if mods and not self.chord_options(item[0], mods[0]):
            raise UnreachableChord(f"{mods[0]}+{item[0]}: no thorax pose within the thorax bounds puts a foreleg on each "
                                   f"key (press points {self.chord_span(item[0], mods[0]):.1f} key pitches apart; two "
                                   f"forelegs span at most about 11)")

    def _chord_plan(self, item) -> dict:
        """Modifier leg, key leg and thorax goal for ``item``; keeps a modifier that a leg already holds if possible."""
        kb = self.sim.keyboard
        index = self.qi if self.queue is not None else len(self.buffer)
        signature = (index, tuple(sorted(kb.down.items())))
        if self._chord_plan_cache is not None and self._chord_plan_cache[0] == signature:
            return self._chord_plan_cache[1]
        key_id, mod = item[0], self._hold_mods(item)[0]
        options = self.chord_options(key_id, mod)
        if not options:
            self.check_chord(item)
        here = self.body[:2]
        best = None
        for ratio, mod_id, mod_leg, key_leg, xy in options:
            if kb.holding(mod_leg) == mod_id:      # already down: stay put if both keys are still reachable
                now = self._pair_ratio(key_id, mod_id, mod_leg, key_leg, here)
                if now <= 0.95 and (best is None or now < best[0]):
                    best = (now, mod_id, mod_leg, key_leg, here.copy())
        if best is None:
            best = min(options, key=lambda o: float(np.linalg.norm(o[4] - here)) + 1.5 * o[0])
        plan = {"mod": best[1], "mod_leg": best[2], "key_leg": best[3], "goal": best[4]}
        self._chord_plan_cache = (signature, plan)
        return plan
''')

rep('''        if self.queue is not None:
            if self.qi >= len(self.queue):
                return None
            return physical_key_id(*self.queue[self.qi], self.sim.keyboard.shift_latched)
        if len(self.buffer) >= len(self.target):
            return None
        return physical_key(self.target[len(self.buffer)], self.sim.keyboard.shift_latched)''','''        if self.queue is not None or self.sim.params.shift_mode == "held":
            item = self._current_item()
            if item is None:
                return None
            if self._fn_tap_pending(item):
                return "Fn"
            if item[1] and self.sim.params.shift_mode == "held":
                return item[0]           # held Shift: the chord's key; the Shift foreleg is the expert's business
            return physical_key_id(item[0], item[1], self.sim.keyboard.shift_latched)
        if len(self.buffer) >= len(self.target):
            return None
        return physical_key(self.target[len(self.buffer)], self.sim.keyboard.shift_latched)''')

rep('''        self.tick, self.last_correct, self.tip_vz = 0, 0.0, np.zeros(len(LEGS))''','''        self._chord_plan_cache = None
        self.tick, self.last_correct, self.tip_vz = 0, 0.0, np.zeros(len(LEGS))''')

# step: wanted modifiers
rep('''        shift_needed = bool(self.next_key()) and self.next_key() in {"ShiftLeft", "ShiftRight"}
''','''        shift_needed = bool(self.next_key()) and self.next_key() in {"ShiftLeft", "ShiftRight"}
        wanted: set[str] = set()      # modifier keys the current press wants held (chords)
        item = self._current_item()
        if item is not None:
            for spec in self._hold_mods(item):
                wanted.update(SHIFT_IDS if spec == "Shift" else (spec,))
''')
rep('''                reward += w["shift"] if shift_needed else -w["shift"]''','''                reward += w["shift"] if (shift_needed or event["key_id"] in wanted) else -w["shift"]''')
rep('''                good = want is not None and event["key_id"] == want[0] and want[1] == any(
                    m in ("ShiftLeft", "ShiftRight") for m in event.get("modifiers", []))''','''                mods = event.get("modifiers", [])
                good = want is not None and event["key_id"] == want[0] and want[1] == any(
                    m in ("ShiftLeft", "ShiftRight") for m in mods) and ("Fn" in want[2]) == ("Fn" in mods)''')

# expert
rep('''        typer = self.typing_leg()
        busy = False
        for k, leg in enumerate(TYPERS):
            block = slice(2 + 3 * k, 5 + 3 * k)''','''        chord = self._current_chord()
        if chord is not None:
            return self._expert_output(self._chord_expert(chord))
        typer = self.typing_leg()
        busy = False
        for k, leg in enumerate(TYPERS):
            block = slice(2 + 3 * k, 5 + 3 * k)''')

rep('''    def _expert_output(self, claw''','''    def _strike_z(self, key_id: str, tip: np.ndarray) -> float:
        """Claw height action for striking ``key_id``: hover near or off the edge, strike well inside, keep pushing."""
        lx, ly = to_layout(float(tip[0]), float(tip[1]))
        key = BY_ID[key_id]
        margin = min(lx - key.x, key.x + key.width - lx, ly - key.y, key.y + key.height - ly)
        margin -= 2.0 * float(np.linalg.norm(self.velocity))
        pushing = self.sim.depression()[self.sim.key_ids.index(key_id)] > 0.005 and margin > 0.05
        return -1.0 if pushing else 1.0 - 2.0 * float(np.clip((margin - 0.1) / 0.15, 0.0, 1.0))

    def _chord_expert(self, item) -> np.ndarray:
        """Two-foreleg chord: walk to a pose where both forelegs reach, press and hold the modifier with one, strike
        the key with the other, then lift both (the modifier stays down when the next press shares it and is still
        reachable, which is how a Shift selection keeps Shift held across several arrows)."""
        plan = self._chord_plan(item)
        key_id, mod_id, mod_leg, key_leg = item[0], plan["mod"], plan["mod_leg"], plan["key_leg"]
        kb = self.sim.keyboard
        holding_mod = kb.holding(mod_leg) == mod_id
        goal_offset = plan["goal"] - self.body[:2]
        settled = holding_mod or float(np.linalg.norm(goal_offset)) < 0.12
        action = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        prev = self._prev()
        busy = False
        for k, leg in enumerate(TYPERS):
            block = slice(2 + 3 * k, 5 + 3 * k)
            tip = self.tips[LEGS.index(leg)]
            target = mod_id if leg == mod_leg else key_id
            over = hit_test(*to_layout(float(tip[0]), float(tip[1]))) == target
            if not (leg == mod_leg and holding_mod) and (kb.holding(leg) is not None or (tip[2] < 0.5 * HOVER and not over)):
                action[block] = [*prev[block][:2], 1.0]      # just pressed (or a stray touch): lift in place
                busy = True
                continue
            offset = press_point(target, float(self.anchor_world(leg)[0])) - self.anchor_world(leg)
            action[block][:2] = np.clip(offset / REACH_XY, -1, 1)
            if leg == mod_leg:
                action[block.start + 2] = -1.0 if holding_mod else (self._strike_z(target, tip) if settled else 1.0)
            else:
                action[block.start + 2] = self._strike_z(target, tip) if (holding_mod and settled) else 1.0
        if busy or holding_mod:
            action[:2] = 0.0
        elif float(np.linalg.norm(goal_offset)) > 0.08:
            action[:2] = np.clip(goal_offset / self.max_speed * 0.8, -1, 1)
        return action

    def _expert_output(self, claw''')
open(p,'w').write(s)
