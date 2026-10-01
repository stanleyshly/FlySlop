import unittest

from backend import exec_judge as ej

SV_OK = "module m(input logic a, output logic y); assign y = ~a; endmodule\n"
SV_TB = """module tb;
  logic a, y;
  m dut(.a(a), .y(y));
  initial begin
    a = 0; #1; if (y !== 1) $display("FAIL a=0");
    a = 1; #1; if (y !== 0) $display("FAIL a=1");
    $display("Mismatches: 0 in 2 samples");
    $finish;
  end
endmodule
"""


class PythonJudgeTests(unittest.TestCase):
    def test_pass(self):
        r = ej.run_python("def f(x):\n    return x + 1\n", "assert f(1) == 2\nprint('ok')")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["stage"], "test")
        self.assertIn("ok", r["stdout_tail"])
        self.assertGreater(r["duration"], 0)

    def test_fail(self):
        r = ej.run_python("def f(x):\n    return x\n", "assert f(1) == 2")
        self.assertFalse(r["ok"])
        self.assertIn("AssertionError", r["error"])

    def test_syntax_error_is_parse_stage(self):
        r = ej.run_python("def f(:\n", "")
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "parse")

    def test_timeout(self):
        r = ej.run_python("while True:\n    pass\n", "", timeout=1.0)
        self.assertFalse(r["ok"])
        self.assertIn("timeout", r["error"])
        self.assertLess(r["duration"], 5)

    def test_network_blocked(self):
        code = ("import socket\n"
                "try:\n    socket.create_connection(('example.com', 80), timeout=2)\n"
                "except OSError as e:\n    print('blocked:', e)\n"
                "else:\n    raise SystemExit('network reachable')\n"
                "try:\n    socket.socket()\nexcept OSError:\n    print('socket blocked')\n")
        r = ej.run_python(code, "")
        self.assertTrue(r["ok"], r)
        self.assertIn("blocked", r["stdout_tail"])
        self.assertIn("socket blocked", r["stdout_tail"])

    def test_temp_cwd(self):
        r = ej.run_python("import os\nprint(os.getcwd())\n")
        self.assertIn("flyslop-exec-", r["stdout_tail"])


class SvJudgeTests(unittest.TestCase):
    def test_parse_fail(self):
        r = ej.run_sv("module m(input a; endmodule\n")
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "parse")

    def test_elab_fail(self):
        r = ej.run_sv("module m(output logic y); assign y = undefined_sig; endmodule\n")
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "elab")

    def test_elab_pass_without_tests(self):
        r = ej.run_sv(SV_OK)
        self.assertTrue(r["ok"])
        self.assertEqual(r["stage"], "elab")

    def test_simulation(self):
        tools = ej.available_tools()
        if not (tools["iverilog"] or tools["verilator"]):
            self.skipTest("no SV simulator installed")
        good = ej.run_sv(SV_OK, SV_TB, timeout=120)
        self.assertTrue(good["ok"], good)
        self.assertEqual(good["stage"], "test")
        bad = ej.run_sv(SV_OK.replace("~a", "a"), SV_TB, timeout=120)
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["stage"], "test")


if __name__ == "__main__":
    unittest.main()
