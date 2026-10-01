import random
import statistics
import unittest

from backend.editor import Editor
from backend.error_inject import NEIGHBOURS, corrupt_buffer, corrupt_keys
from backend.keyboard import BY_ID
from backend.keyplan import (apply_commands, apply_tokens, diff_size, expand_tokens, keystroke_cost,
                             naive_cost, oracle_edits, oracle_plan, text_to_keys)

SNIPPETS = ["def add(a, b):\n    return a + b\n", "for i in range(10):\n    print(i)\n", "x = {'k': [1, 2, 3]}\n",
            "class A:\n    def f(self):\n        return self.x % 3 == 0\n", "if a<b and c>=d:\n    y = ~x | z & 1\n",
            "s = \"Hello, World!\"  # TODO: fix\n", "import os\nimport sys\n\nprint(os.path.join('a', 'b'))\n"]


def mutate(rng: random.Random, text: str) -> str:
    chars = list(text)
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["ins", "del", "sub", "line", "indent"])
        i = rng.randrange(len(chars) + 1)
        if kind == "ins":
            chars[i:i] = list(rng.choice(["foo", "X", "(a, b)", "\n    ", "# hi!", "    ", "{}", "\n"]))
        elif kind == "del" and chars:
            del chars[i:i + rng.randint(1, 12)]
        elif kind == "sub" and chars:
            chars[min(i, len(chars) - 1)] = rng.choice("@#$%^&*()_+{}|:\"<>?~AZaz09 ")
        elif kind == "line":
            chars[i:i] = list(rng.choice(SNIPPETS))
        else:
            chars[i:i] = list("    ")
    return "".join(chars)


class ExpandTests(unittest.TestCase):
    def test_specials_and_shift(self):
        cmds = expand_tokens(["<SEL_START>", "<RIGHT>", "<SEL_END>", "<HOME>", "<DEL>", "<ENTER>", "<TAB>", "<EOS>"])
        self.assertEqual([c.key for c in cmds], ["ShiftLeft", "ArrowRight", "ShiftLeft", "ArrowLeft", "Backspace", "Enter", "Tab"])
        self.assertTrue(cmds[0].hold and cmds[2].release and cmds[3].mods == ("Fn",))
        self.assertEqual(cmds[0].as_dict(), {"key": "ShiftLeft", "mods": [], "hold": True})
        self.assertEqual(cmds[2].as_dict(), {"key": "ShiftLeft", "release": True})
        self.assertEqual(text_to_keys("A")[0].as_dict(), {"key": "a", "mods": ["ShiftLeft"], "hold": False})
        self.assertEqual(text_to_keys("!")[0].key, "1")
        self.assertEqual(text_to_keys(" ")[0].key, "Space")

    def test_ids_and_tokenizer(self):
        class Tok:
            def decode_token(self, i):
                return {40: "Hi", 41: "_(x)"}[i]
        ed = apply_tokens(Editor(""), [40, 4, 41, 1, 2], Tok())
        self.assertEqual(ed.text, "Hi\n_(x)")

    def test_selection_bracketing(self):
        ed = Editor("abcdef", 1, 1)
        apply_tokens(ed, ["<SEL_START>", "<RIGHT>", "<RIGHT>", "<SEL_END>"])
        self.assertEqual((ed.anchor, ed.cursor), (1, 3))
        apply_tokens(ed, ["X"])
        self.assertEqual(ed.text, "aXdef")

    def test_roundtrip_text(self):
        rng = random.Random(1)
        for _ in range(100):
            text = "".join(rng.choice(SNIPPETS) for _ in range(2))
            self.assertEqual(apply_commands(Editor(""), text_to_keys(text)) and Editor("").text, "")
            ed = Editor("")
            apply_commands(ed, text_to_keys(text))
            self.assertEqual(ed.text, text)
            ed2 = Editor("")
            from backend.keyplan import _type_tokens
            apply_tokens(ed2, _type_tokens(text))
            self.assertEqual(ed2.text, text)


class OracleTests(unittest.TestCase):
    def test_property_and_cost(self):
        rng = random.Random(7)
        ratios, vs_diff = [], []
        for n in range(600):
            buf = "".join(rng.choice(SNIPPETS) for _ in range(rng.randint(1, 3)))
            target = mutate(rng, buf) if rng.random() < 0.9 else rng.choice(SNIPPETS)
            cursor = rng.randint(0, len(buf))
            sel = None
            if rng.random() < 0.3:
                anchor = rng.randint(0, len(buf))
                sel = {"anchor": anchor, "head": cursor} if anchor != cursor else None
            tokens, cost = oracle_plan(buf, cursor, sel, target)
            ed = Editor(buf, cursor, sel["anchor"] if sel else cursor)
            apply_tokens(ed, tokens)
            self.assertEqual(ed.text, target, (n, buf, target, tokens))
            self.assertEqual(cost, keystroke_cost(tokens))
            self.assertLessEqual(cost, naive_cost(buf, target) + len(buf) + 4)
            if buf != target:
                ratios.append(cost / naive_cost(buf, target))
                vs_diff.append(cost / max(1, diff_size(buf, target)))
        print(f"\noracle cost: mean ratio vs naive {statistics.mean(ratios):.3f}, "
              f"vs diff size {statistics.mean(vs_diff):.2f} (median {statistics.median(vs_diff):.2f}), n={len(ratios)}")
        self.assertLess(statistics.mean(ratios), 0.5)

    def test_edge_cases(self):
        self.assertEqual(oracle_edits("abc", 3, None, "abc"), [])
        tokens = oracle_edits("abc", 1, {"anchor": 0, "head": 1}, "abc")   # stray selection only
        ed = Editor("abc", 1, 0)
        apply_tokens(ed, tokens)
        self.assertEqual((ed.text, ed.anchor == ed.cursor), ("abc", True))
        for buf, tgt in [("", "a\nB"), ("abc", ""), ("a\n\nb", "a\nb\n")]:
            ed = Editor(buf, 0, 0)
            apply_tokens(ed, oracle_edits(buf, 0, None, tgt))
            self.assertEqual(ed.text, tgt)

    def test_selection_is_used(self):
        buf = "x = 1\nyyyyyyyyyyyyyyyyyyyy\nz = 2"
        tokens = oracle_edits(buf, 6, None, "x = 1\nq\nz = 2")
        self.assertLess(keystroke_cost(tokens), 20)
        self.assertIn("<SEL_START>", tokens)


class InjectTests(unittest.TestCase):
    def test_buffer_deterministic(self):
        text = "def f(x):\n    return x + 1\n"
        a, b = corrupt_buffer(text, 5, seed=3, n=3), corrupt_buffer(text, 5, seed=3, n=3)
        self.assertEqual(a, b)
        self.assertNotEqual(a.text + str(a.cursor), text + "5")
        self.assertEqual(len(a.errors), 3)
        for s in range(50):
            c = corrupt_buffer(text, 5, seed=s, n=2)
            self.assertTrue(0 <= c.cursor <= len(c.text))

    def test_key_corruptor(self):
        cmds = text_to_keys("hello world; print(1)\n" * 5)
        a = corrupt_keys(cmds, 0.2, seed=9)
        self.assertEqual(a, corrupt_keys(cmds, 0.2, seed=9))
        self.assertNotEqual(a, corrupt_keys(cmds, 0.2, seed=10))
        self.assertEqual(corrupt_keys(cmds, 0.0, seed=9)[0], cmds)
        out, slipped = a
        self.assertTrue(slipped)
        for i in slipped:
            self.assertIn(out[i].key, NEIGHBOURS[cmds[i].key])
            self.assertEqual(out[i].mods, cmds[i].mods)
        self.assertLess(len(slipped) / len(cmds), 0.4)
        for c in out:
            self.assertIn(c.key, BY_ID)

    def test_modifiers_untouched(self):
        cmds = expand_tokens(["<SEL_START>", "<RIGHT>", "<SEL_END>"])
        self.assertEqual(corrupt_keys(cmds, 1.0, seed=1)[0][0], cmds[0])


if __name__ == "__main__":
    unittest.main()
