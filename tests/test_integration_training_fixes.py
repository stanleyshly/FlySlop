"""Integration fixes: selfcorrect start-pass, eval tokenizer/checkpoint keys, oracle_gen near-dup flags."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import eval_microsuite as em
from training import oracle_gen as og
from training.selfcorrect_env import SelfCorrectEnv
from training.stages import stage5_oracle_data as s5
from training.tokenizer import BPETokenizer

from tests.test_oracle_gen import GOOD_PY as GOOD

TASK = {"id": "t", "lang": "py", "tests": "", "prompt": "identity"}


def ok_judge(lang, code, tests="", timeout=0.0, **kw):
    return {"ok": True, "stage": "pass", "duration": 0.0}


class StartPass(unittest.TestCase):
    def test_passing_start_buffer_terminates(self):
        env = SelfCorrectEnv(judge=ok_judge)
        env.reset(TASK, start_buffer=GOOD)
        self.assertTrue(env.done)
        self.assertEqual(env.reason, "pass")
        self.assertEqual(env.reward, 1.0)
        self.assertEqual(len(env.runs), 1)

    def test_failing_start_continues(self):
        env = SelfCorrectEnv(judge=lambda *a, **k: {"ok": False, "stage": "test", "duration": 0.0})
        env.reset(TASK, start_buffer="x")
        self.assertFalse(env.done)

    def test_run_episode_returns_on_pass(self):
        env = SelfCorrectEnv(judge=ok_judge)
        info = env.run_episode(em.NullPolicy(), TASK, start_buffer=GOOD)
        self.assertEqual(info["id"], "t")


class EvalMicrosuite(unittest.TestCase):
    def test_default_tokenizer_loads_bpe_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bpe.json"
            BPETokenizer.train(["hello hello hello world world"] * 5, vocab_size=300, min_count=2).save(p)
            self.assertGreater(em.default_tokenizer(p).vocab_size, BPETokenizer().vocab_size)
            self.assertEqual(em.default_tokenizer(Path(d) / "none.json").vocab_size, BPETokenizer().vocab_size)

    def test_load_policy_accepts_model_and_state_dict_keys(self):
        import torch
        for key in ("model", "state_dict"):
            fake_model = mock.MagicMock()
            fake_model.eval.return_value = fake_model
            fake_mod = mock.MagicMock(build_model=mock.MagicMock(return_value=fake_model))
            with mock.patch.object(torch, "load", return_value={"kind": "k", "cfg": None, key: {"w": 1}}), \
                    mock.patch.dict(sys.modules, {"backend.connectome.thinker": fake_mod}):
                em.load_policy("thinker:x.pt")
            fake_model.load_state_dict.assert_called_once_with({"w": 1})


class OracleNeardup(unittest.TestCase):
    def test_cli_flags_and_filter_thresholds(self):
        f = og.Filters(judge=ok_judge, neardup_accepted=1.01)   # never triggers
        raw = "```python\n" + GOOD + "```"
        self.assertTrue(f.run(raw, {"lang": "py"})["passed"])
        f.register(GOOD, "py")
        g = og.Filters([{"lang": "py", "code": GOOD}], judge=ok_judge, neardup_test=1.01)
        self.assertTrue(g.run(raw, {"lang": "py"})["passed"])
        self.assertEqual(og.Filters([{"lang": "py", "code": GOOD}], judge=ok_judge).run(raw, {"lang": "py"})["fail_stage"], "neardup")

    def test_stage_passes_flags(self):
        o = {"run": "r", "seed": 0, "spec_tokens": 1, "code_tokens": 2, "standin": False, "limit": None,
             "sources": None, "holdout_sources": None, "neardup_test": 0.5, "neardup_accepted": 0.8}
        cmd = s5.build_cmd(o, None, None)
        self.assertEqual(cmd[cmd.index("--neardup-test") + 1], "0.5")
        self.assertEqual(cmd[cmd.index("--neardup-accepted") + 1], "0.8")


if __name__ == "__main__":
    unittest.main()
