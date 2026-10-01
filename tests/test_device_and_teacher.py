import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import memory_budget as mb  # noqa: E402
from backend.connectome import device as dev  # noqa: E402


class DeviceTests(unittest.TestCase):
    def bench(self, best):
        d = tempfile.mkdtemp()
        p = Path(d) / "b.json"
        p.write_text(json.dumps({"best_train": best}))
        dev.load_bench.cache_clear()
        return p

    def test_cpu_and_invalid(self):
        self.assertEqual(dev.resolve_device("cpu"), "cpu")
        with self.assertRaises(ValueError):
            dev.resolve_device("cuda")

    def test_auto_missing_cache_is_cpu(self):
        self.assertEqual(dev.resolve_device("auto", 4000, path=Path("/nonexistent/x.json")), "cpu")

    def test_auto_follows_cache(self):
        p = self.bench({"4k": {"device": "mps", "backend": "dense"}, "20k": {"device": "cpu", "backend": "csr_fn"}})
        with mock.patch.object(dev, "mps_available", return_value=True):
            self.assertEqual(dev.resolve_device("auto", 4000, path=p), "mps")
            self.assertEqual(dev.resolve_device("auto", 20000, path=p), "cpu")
        with mock.patch.object(dev, "mps_available", return_value=False):
            self.assertEqual(dev.resolve_device("auto", 4000, path=p), "cpu")
            self.assertEqual(dev.resolve_device("mps"), "cpu")

    def test_shipped_cache_recommends_cpu_sparse(self):
        dev.load_bench.cache_clear()
        if dev.load_bench() is None:
            self.skipTest("no device_bench.json yet")
        self.assertEqual(dev.recommended(20000)["device"], "cpu")


class MemoryBudgetTests(unittest.TestCase):
    def test_cap_env(self):
        with mock.patch.dict(os.environ, {mb.ENV: "2.5"}):
            self.assertEqual(mb.max_ram_bytes(), int(2.5 * mb.GB))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(mb.ENV, None)
            self.assertEqual(mb.max_ram_bytes(), 4 * mb.GB)
            self.assertEqual(mb.max_ram_bytes(config_gb=1.0), mb.GB)

    def test_check_fits(self):
        with mock.patch.dict(os.environ, {mb.ENV: "4"}):
            self.assertTrue(mb.check_fits(1024, "tiny"))
            self.assertFalse(mb.check_fits(8 * mb.GB, "huge"))
            with self.assertRaises(mb.OverBudget):
                mb.check_fits(8 * mb.GB, "huge", raise_error=True)

    def test_rss_and_workers(self):
        self.assertGreater(mb.current_rss(), 1_000_000)
        self.assertGreaterEqual(mb.current_rss(tree=True), mb.current_rss())
        self.assertGreater(mb.peak_rss(), 0)
        with mock.patch.dict(os.environ, {mb.ENV: "4"}):
            self.assertEqual(mb.worker_budget(int(0.5 * mb.GB), reserve_bytes=mb.GB), 6)
            self.assertEqual(mb.worker_budget(int(0.5 * mb.GB), mb.GB, max_workers=3), 3)
            self.assertEqual(mb.worker_budget(5 * mb.GB), 0)


class CsrBackendTests(unittest.TestCase):
    def test_csr_fn_matches_index_add(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        spec = importlib.util.spec_from_file_location("bench_device", ROOT / "scripts" / "bench_device.py")
        bd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bd)
        n, batch = 50, 3
        g = torch.Generator().manual_seed(1)
        dst, src, val = bd.make_edges(n, 6, g)
        x = torch.randn(batch, n, generator=g)
        grads = {}
        for name in ("csr_fn", "index_add"):
            step, params = bd.build(name, n, dst, src, val, torch.device("cpu"))
            r = torch.zeros(batch, n)
            for _ in range(3):
                r = step(r, x)
            r.square().sum().backward()
            grads[name] = (r.detach(), params[0].grad.clone())
        self.assertTrue(torch.allclose(grads["csr_fn"][0], grads["index_add"][0], atol=1e-5))
        # csr_fn stores values sorted by (dst, src); compare as multisets (duplicate edges accumulate identically)
        self.assertAlmostEqual(float(grads["csr_fn"][1].sum()), float(grads["index_add"][1].sum()), places=3)
        self.assertAlmostEqual(float(grads["csr_fn"][1].abs().sum()), float(grads["index_add"][1].abs().sum()), places=3)


class TeacherTests(unittest.TestCase):
    def test_import_does_not_load_model(self):
        from training import teacher
        t = teacher.Teacher()
        self.assertIsNone(t._model)
        import subprocess
        code = "import sys; import training.teacher as t; t.Teacher(); assert 'mlx_lm' not in sys.modules and 'mlx' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
        self.assertEqual(t.model_id, teacher.TEACHER_MODEL)
        with self.assertRaises(ValueError):
            teacher.Teacher(backend="nope")

    def test_footprint_estimate_and_refusal(self):
        from training import teacher
        self.assertLess(teacher.estimate_model_bytes(teacher.TEACHER_MODEL), 4 * mb.GB)
        with mock.patch.dict(os.environ, {mb.ENV: "1"}):
            with self.assertRaises(mb.OverBudget):
                teacher.Teacher(teacher.TEACHER_MODEL).load()

    @unittest.skipUnless(os.environ.get("FLYSLOP_TEST_TEACHER") == "1", "set FLYSLOP_TEST_TEACHER=1")
    def test_generate_standin(self):
        from training import teacher
        t = teacher.Teacher(teacher.STANDIN_MODEL, max_tokens=24)
        out = t.generate_batch(["Write def double(x):"])
        self.assertTrue(out[0].strip())
        self.assertGreater(t.last_stats["tokens_per_s"], 0)


if __name__ == "__main__":
    unittest.main()
