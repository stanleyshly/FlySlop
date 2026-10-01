"""Original (MIT, FlySlop-authored) SystemVerilog micro-programs with self-checking testbenches.

Combinational testbenches are generated exhaustively from a reference expression; sequential ones are
written out. Every testbench ends with ``Mismatches: N in M samples`` (the exec_judge pass marker)."""
_T = []


def _w(n):
    return f"[{n - 1}:0] " if n > 1 else ""


def _comb_tb(mod, ins, out, expr):
    n = sum(w for _, w in ins)
    decl = "\n".join(f"  logic {_w(w)}{nm};" for nm, w in ins)
    conns = ", ".join(f".{nm}({nm})" for nm, _ in ins + [out])
    cat = ", ".join(nm for nm, _ in ins)
    return f"""module tb;
{decl}
  wire {_w(out[1])}{out[0]};
  logic {_w(out[1])}expv;
  integer i, bad;
  {mod} dut({conns});
  initial begin
    bad = 0;
    for (i = 0; i < {2 ** n}; i = i + 1) begin
      {{{cat}}} = i[{n - 1}:0];
      #1;
      expv = {expr};
      if ({out[0]} !== expv) bad = bad + 1;
    end
    $display("Mismatches: %0d in %0d samples", bad, {2 ** n});
    $finish;
  end
endmodule
"""


def _comb(id, diff, prompt, code, ins, out, expr):
    mod = id
    _T.append({"id": f"sv/{id}", "lang": "sv", "difficulty": diff, "prompt": prompt,
               "code": code.strip("\n") + "\n", "tests": _comb_tb(mod, ins, out, expr)})


def _seq(id, diff, prompt, code, tests):
    _T.append({"id": f"sv/{id}", "lang": "sv", "difficulty": diff, "prompt": prompt,
               "code": code.strip("\n") + "\n", "tests": tests.strip("\n") + "\n"})


A, B, Y = ("a", 1), ("b", 1), ("y", 1)
_comb("passthru", 1, "Write a SystemVerilog module passthru with input a and output y that connects y directly to a.",
      "module passthru(input logic a, output logic y);\n  assign y = a;\nendmodule", [A], Y, "a")
_comb("inv1", 1, "Write a module inv1 with input a and output y that inverts a.",
      "module inv1(input logic a, output logic y);\n  assign y = ~a;\nendmodule", [A], Y, "~a")
_comb("and2", 1, "Write a module and2 with inputs a, b and output y that is the AND of a and b.",
      "module and2(input logic a, input logic b, output logic y);\n  assign y = a & b;\nendmodule", [A, B], Y, "a & b")
_comb("or2", 1, "Write a module or2 with inputs a, b and output y that is the OR of a and b.",
      "module or2(input logic a, input logic b, output logic y);\n  assign y = a | b;\nendmodule", [A, B], Y, "a | b")
_comb("xor2", 1, "Write a module xor2 with inputs a, b and output y that is the XOR of a and b.",
      "module xor2(input logic a, input logic b, output logic y);\n  assign y = a ^ b;\nendmodule", [A, B], Y, "a ^ b")
_comb("nand2", 1, "Write a module nand2 with inputs a, b and output y that is the NAND of a and b.",
      "module nand2(input logic a, input logic b, output logic y);\n  assign y = ~(a & b);\nendmodule", [A, B], Y, "~(a & b)")
_comb("nor2", 1, "Write a module nor2 with inputs a, b and output y that is the NOR of a and b.",
      "module nor2(input logic a, input logic b, output logic y);\n  assign y = ~(a | b);\nendmodule", [A, B], Y, "~(a | b)")
_comb("xnor2", 1, "Write a module xnor2 with inputs a, b and output y that is the XNOR of a and b.",
      "module xnor2(input logic a, input logic b, output logic y);\n  assign y = ~(a ^ b);\nendmodule", [A, B], Y, "~(a ^ b)")
_comb("and3", 1, "Write a module and3 with inputs a, b, c and output y that is the AND of the three inputs.",
      "module and3(input logic a, input logic b, input logic c, output logic y);\n  assign y = a & b & c;\nendmodule",
      [A, B, ("c", 1)], Y, "a & b & c")
_comb("maj3", 2, "Write a module maj3 with inputs a, b, c and output y that is 1 when at least two inputs are 1.",
      "module maj3(input logic a, input logic b, input logic c, output logic y);\n  assign y = (a & b) | (a & c) | (b & c);\nendmodule",
      [A, B, ("c", 1)], Y, "(a & b) | (a & c) | (b & c)")
_comb("mux2", 2, "Write a module mux2 with inputs a, b, sel and output y. y is b when sel is 1, otherwise a.",
      "module mux2(input logic a, input logic b, input logic sel, output logic y);\n  assign y = sel ? b : a;\nendmodule",
      [A, B, ("sel", 1)], Y, "sel ? b : a")
_comb("mux2w4", 2, "Write a module mux2w4 with 4-bit inputs a, b, a 1-bit input sel and 4-bit output y. y is b when sel is 1, otherwise a.",
      "module mux2w4(input logic [3:0] a, input logic [3:0] b, input logic sel, output logic [3:0] y);\n  assign y = sel ? b : a;\nendmodule",
      [("a", 4), ("b", 4), ("sel", 1)], ("y", 4), "sel ? b : a")
_comb("mux4", 3, "Write a module mux4 with 1-bit inputs d0, d1, d2, d3, a 2-bit input sel and output y that selects d[sel].",
      "module mux4(input logic d0, input logic d1, input logic d2, input logic d3, input logic [1:0] sel, output logic y);\n"
      "  always_comb begin\n    case (sel)\n      2'd0: y = d0;\n      2'd1: y = d1;\n      2'd2: y = d2;\n      default: y = d3;\n    endcase\n  end\nendmodule",
      [("d0", 1), ("d1", 1), ("d2", 1), ("d3", 1), ("sel", 2)], Y, "sel == 0 ? d0 : sel == 1 ? d1 : sel == 2 ? d2 : d3")
_comb("half_add", 2, "Write a module half_add with inputs a, b and outputs s (sum, 1 bit) packed as a 2-bit output y where y[1] is carry and y[0] is sum.",
      "module half_add(input logic a, input logic b, output logic [1:0] y);\n  assign y[0] = a ^ b;\n  assign y[1] = a & b;\nendmodule",
      [A, B], ("y", 2), "a + b")
_comb("full_add", 2, "Write a module full_add with 1-bit inputs a, b, cin and a 2-bit output y equal to a + b + cin (y[1] is carry out).",
      "module full_add(input logic a, input logic b, input logic cin, output logic [1:0] y);\n  assign y = a + b + cin;\nendmodule",
      [A, B, ("cin", 1)], ("y", 2), "a + b + cin")
_comb("add4", 2, "Write a module add4 with 4-bit inputs a, b and a 5-bit output y equal to a + b.",
      "module add4(input logic [3:0] a, input logic [3:0] b, output logic [4:0] y);\n  assign y = {1'b0, a} + {1'b0, b};\nendmodule",
      [("a", 4), ("b", 4)], ("y", 5), "a + b")
_comb("sub4", 2, "Write a module sub4 with 4-bit inputs a, b and a 4-bit output y equal to a minus b modulo 16.",
      "module sub4(input logic [3:0] a, input logic [3:0] b, output logic [3:0] y);\n  assign y = a - b;\nendmodule",
      [("a", 4), ("b", 4)], ("y", 4), "a - b")
_comb("eq4", 2, "Write a module eq4 with 4-bit inputs a, b and a 1-bit output y that is 1 when a equals b.",
      "module eq4(input logic [3:0] a, input logic [3:0] b, output logic y);\n  assign y = (a == b);\nendmodule",
      [("a", 4), ("b", 4)], Y, "(a == b)")
_comb("gt4", 2, "Write a module gt4 with 4-bit inputs a, b and a 1-bit output y that is 1 when a is greater than b.",
      "module gt4(input logic [3:0] a, input logic [3:0] b, output logic y);\n  assign y = (a > b);\nendmodule",
      [("a", 4), ("b", 4)], Y, "(a > b)")
_comb("parity4", 2, "Write a module parity4 with a 4-bit input a and output y that is 1 when a has an odd number of ones.",
      "module parity4(input logic [3:0] a, output logic y);\n  assign y = ^a;\nendmodule", [("a", 4)], Y, "^a")
_comb("dec2to4", 3, "Write a module dec2to4 with a 2-bit input a and a 4-bit output y where bit a of y is 1 and the others 0.",
      "module dec2to4(input logic [1:0] a, output logic [3:0] y);\n  always_comb begin\n    case (a)\n      2'd0: y = 4'b0001;\n      2'd1: y = 4'b0010;\n      2'd2: y = 4'b0100;\n      default: y = 4'b1000;\n    endcase\n  end\nendmodule",
      [("a", 2)], ("y", 4), "4'b0001 << a")
_comb("enc4", 3, "Write a module enc4 with a 4-bit input a and a 2-bit output y giving the index of the highest set bit of a, 0 if a is zero.",
      "module enc4(input logic [3:0] a, output logic [1:0] y);\n  always_comb begin\n    if (a[3]) y = 2'd3;\n    else if (a[2]) y = 2'd2;\n    else if (a[1]) y = 2'd1;\n    else y = 2'd0;\n  end\nendmodule",
      [("a", 4)], ("y", 2), "a[3] ? 2'd3 : a[2] ? 2'd2 : a[1] ? 2'd1 : 2'd0")
_comb("gray2", 3, "Write a module gray2 with a 2-bit input a and a 2-bit output y that converts binary to Gray code using a case statement.",
      "module gray2(input logic [1:0] a, output logic [1:0] y);\n  always_comb begin\n    case (a)\n      2'b00: y = 2'b00;\n      2'b01: y = 2'b01;\n      2'b10: y = 2'b11;\n      default: y = 2'b10;\n    endcase\n  end\nendmodule",
      [("a", 2)], ("y", 2), "a ^ (a >> 1)")
_comb("popcnt3", 3, "Write a module popcnt3 with a 3-bit input a and a 2-bit output y counting the ones in a.",
      "module popcnt3(input logic [2:0] a, output logic [1:0] y);\n  assign y = a[0] + a[1] + a[2];\nendmodule",
      [("a", 3)], ("y", 2), "a[0] + a[1] + a[2]")
_comb("bitrev4", 2, "Write a module bitrev4 with a 4-bit input a and a 4-bit output y that is a with its bits reversed.",
      "module bitrev4(input logic [3:0] a, output logic [3:0] y);\n  assign y = {a[0], a[1], a[2], a[3]};\nendmodule",
      [("a", 4)], ("y", 4), "{a[0], a[1], a[2], a[3]}")
_comb("shl1", 1, "Write a module shl1 with a 4-bit input a and a 4-bit output y equal to a shifted left by one bit.",
      "module shl1(input logic [3:0] a, output logic [3:0] y);\n  assign y = a << 1;\nendmodule", [("a", 4)], ("y", 4), "a << 1")
_comb("absdiff", 3, "Write a module absdiff with 4-bit inputs a, b and a 4-bit output y equal to the absolute difference of a and b.",
      "module absdiff(input logic [3:0] a, input logic [3:0] b, output logic [3:0] y);\n  always_comb begin\n    if (a > b) y = a - b;\n    else y = b - a;\n  end\nendmodule",
      [("a", 4), ("b", 4)], ("y", 4), "a > b ? a - b : b - a")

_CLK = "  logic clk = 0;\n  always #5 clk = ~clk;\n  integer bad = 0, n = 0;\n"
_END = '  $display("Mismatches: %0d in %0d samples", bad, n);\n    $finish;\n  end\nendmodule\n'

_seq("dff", 2, "Write a module dff with inputs clk, d and output q. q takes d on each rising edge of clk.",
     "module dff(input logic clk, input logic d, output logic q);\n  always_ff @(posedge clk) q <= d;\nendmodule",
     "module tb;\n" + _CLK + """  logic d = 0; wire q;
  dff dut(.clk(clk), .d(d), .q(q));
  integer i;
  initial begin
    for (i = 0; i < 16; i = i + 1) begin
      d = (i * 5 + 3) % 3 == 0;
      @(posedge clk); #1;
      n = n + 1; if (q !== d) bad = bad + 1;
    end
  """ + _END)
_seq("dff_en", 2, "Write a module dff_en with inputs clk, en, d and output q. On a rising clock edge q takes d only when en is 1.",
     "module dff_en(input logic clk, input logic en, input logic d, output logic q);\n  always_ff @(posedge clk) begin\n    if (en) q <= d;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic en = 1, d = 0; wire q; logic ref_q;
  dff_en dut(.clk(clk), .en(en), .d(d), .q(q));
  integer i;
  initial begin
    en = 1; d = 1; @(posedge clk); #1; ref_q = 1;
    for (i = 0; i < 24; i = i + 1) begin
      en = (i % 3) != 0; d = (i % 2) == 0;
      @(posedge clk); #1;
      if (en) ref_q = d;
      n = n + 1; if (q !== ref_q) bad = bad + 1;
    end
  """ + _END)
_seq("dff_arst", 3, "Write a module dff_arst with inputs clk, rst_n, d and output q. An active-low asynchronous reset rst_n clears q to 0, otherwise q takes d on the rising clock edge.",
     "module dff_arst(input logic clk, input logic rst_n, input logic d, output logic q);\n  always_ff @(posedge clk or negedge rst_n) begin\n    if (!rst_n) q <= 1'b0;\n    else q <= d;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst_n = 0, d = 1; wire q;
  dff_arst dut(.clk(clk), .rst_n(rst_n), .d(d), .q(q));
  initial begin
    #1; n = n + 1; if (q !== 1'b0) bad = bad + 1;
    #12 rst_n = 1;
    @(posedge clk); #1; n = n + 1; if (q !== 1'b1) bad = bad + 1;
    d = 0; @(posedge clk); #1; n = n + 1; if (q !== 1'b0) bad = bad + 1;
    d = 1; @(posedge clk); #1; n = n + 1; if (q !== 1'b1) bad = bad + 1;
    #2 rst_n = 0; #1; n = n + 1; if (q !== 1'b0) bad = bad + 1;
  """ + _END)
_seq("counter4", 2, "Write a module counter4 with inputs clk, rst and a 4-bit output q. A synchronous active-high rst sets q to 0, otherwise q increments each rising clock edge.",
     "module counter4(input logic clk, input logic rst, output logic [3:0] q);\n  always_ff @(posedge clk) begin\n    if (rst) q <= 4'd0;\n    else q <= q + 4'd1;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst = 1; wire [3:0] q; logic [3:0] r;
  counter4 dut(.clk(clk), .rst(rst), .q(q));
  integer i;
  initial begin
    @(posedge clk); #1; r = 0; n = n + 1; if (q !== r) bad = bad + 1;
    rst = 0;
    for (i = 0; i < 40; i = i + 1) begin
      if (i == 20) rst = 1; else rst = 0;
      @(posedge clk); #1;
      r = rst ? 4'd0 : r + 4'd1;
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)
_seq("counter_en", 3, "Write a module counter_en with inputs clk, rst, en and an 8-bit output q. Synchronous active-high rst clears q, otherwise q increments on a rising edge only when en is 1.",
     "module counter_en(input logic clk, input logic rst, input logic en, output logic [7:0] q);\n  always_ff @(posedge clk) begin\n    if (rst) q <= 8'd0;\n    else if (en) q <= q + 8'd1;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst = 1, en = 0; wire [7:0] q; logic [7:0] r;
  counter_en dut(.clk(clk), .rst(rst), .en(en), .q(q));
  integer i;
  initial begin
    @(posedge clk); #1; r = 0; n = n + 1; if (q !== r) bad = bad + 1;
    for (i = 0; i < 40; i = i + 1) begin
      rst = (i == 30); en = (i % 4) != 1;
      @(posedge clk); #1;
      if (rst) r = 0; else if (en) r = r + 1;
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)
_seq("updown", 3, "Write a module updown with inputs clk, rst, up and a 4-bit output q. Synchronous rst clears q, otherwise q counts up when up is 1 and down when up is 0.",
     "module updown(input logic clk, input logic rst, input logic up, output logic [3:0] q);\n  always_ff @(posedge clk) begin\n    if (rst) q <= 4'd0;\n    else if (up) q <= q + 4'd1;\n    else q <= q - 4'd1;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst = 1, up = 1; wire [3:0] q; logic [3:0] r;
  updown dut(.clk(clk), .rst(rst), .up(up), .q(q));
  integer i;
  initial begin
    @(posedge clk); #1; r = 0; n = n + 1; if (q !== r) bad = bad + 1;
    for (i = 0; i < 40; i = i + 1) begin
      rst = 0; up = (i % 7) < 4;
      @(posedge clk); #1;
      r = up ? r + 4'd1 : r - 4'd1;
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)
_seq("shreg4", 3, "Write a module shreg4 with inputs clk, din and a 4-bit output q. On each rising edge q shifts left by one and din enters bit 0.",
     "module shreg4(input logic clk, input logic din, output logic [3:0] q);\n  always_ff @(posedge clk) q <= {q[2:0], din};\nendmodule",
     "module tb;\n" + _CLK + """  logic din = 0; wire [3:0] q; logic [3:0] r = 4'b0000;
  shreg4 dut(.clk(clk), .din(din), .q(q));
  integer i;
  initial begin
    for (i = 0; i < 4; i = i + 1) begin din = 1; @(posedge clk); #1; r = {r[2:0], din}; end
    for (i = 0; i < 24; i = i + 1) begin
      din = (i * 7 % 5) < 2;
      @(posedge clk); #1;
      r = {r[2:0], din};
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)
_seq("toggle", 2, "Write a module toggle with inputs clk, rst, t and output q. Synchronous rst clears q; otherwise q flips on a rising edge when t is 1.",
     "module toggle(input logic clk, input logic rst, input logic t, output logic q);\n  always_ff @(posedge clk) begin\n    if (rst) q <= 1'b0;\n    else if (t) q <= ~q;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst = 1, t = 0; wire q; logic r;
  toggle dut(.clk(clk), .rst(rst), .t(t), .q(q));
  integer i;
  initial begin
    @(posedge clk); #1; r = 0; n = n + 1; if (q !== r) bad = bad + 1;
    for (i = 0; i < 30; i = i + 1) begin
      rst = (i == 25); t = (i % 3) != 0;
      @(posedge clk); #1;
      if (rst) r = 0; else if (t) r = ~r;
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)
_seq("reg8", 2, "Write a module reg8 with inputs clk, rst, we, an 8-bit input d and an 8-bit output q. Synchronous rst clears q, otherwise q loads d when we is 1.",
     "module reg8(input logic clk, input logic rst, input logic we, input logic [7:0] d, output logic [7:0] q);\n  always_ff @(posedge clk) begin\n    if (rst) q <= 8'h00;\n    else if (we) q <= d;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst = 1, we = 0; logic [7:0] d = 0; wire [7:0] q; logic [7:0] r;
  reg8 dut(.clk(clk), .rst(rst), .we(we), .d(d), .q(q));
  integer i;
  initial begin
    @(posedge clk); #1; r = 0; n = n + 1; if (q !== r) bad = bad + 1;
    for (i = 0; i < 30; i = i + 1) begin
      rst = (i == 20); we = (i % 2) == 0; d = i * 37 + 11;
      @(posedge clk); #1;
      if (rst) r = 0; else if (we) r = d;
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)
_seq("edge_det", 3, "Write a module edge_det with inputs clk, x and output y. y is 1 for the clock cycle after x rose (x is 1 now and was 0 on the previous rising edge), registered on clk.",
     "module edge_det(input logic clk, input logic x, output logic y);\n  logic prev = 1'b0;\n  always_ff @(posedge clk) begin\n    y <= x & ~prev;\n    prev <= x;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic x = 0; wire y; logic p = 0; logic r = 0;
  edge_det dut(.clk(clk), .x(x), .y(y));
  integer i;
  initial begin
    for (i = 0; i < 30; i = i + 1) begin
      x = ((i * 5) % 7) < 3;
      @(posedge clk); #1;
      r = x & ~p; p = x;
      if (i > 0) begin n = n + 1; if (y !== r) bad = bad + 1; end
    end
  """ + _END)
_seq("mod5", 3, "Write a module mod5 with inputs clk, rst and a 3-bit output q. Synchronous rst clears q; otherwise q counts 0,1,2,3,4 and wraps back to 0.",
     "module mod5(input logic clk, input logic rst, output logic [2:0] q);\n  always_ff @(posedge clk) begin\n    if (rst) q <= 3'd0;\n    else if (q == 3'd4) q <= 3'd0;\n    else q <= q + 3'd1;\n  end\nendmodule",
     "module tb;\n" + _CLK + """  logic rst = 1; wire [2:0] q; logic [2:0] r;
  mod5 dut(.clk(clk), .rst(rst), .q(q));
  integer i;
  initial begin
    @(posedge clk); #1; r = 0; n = n + 1; if (q !== r) bad = bad + 1;
    for (i = 0; i < 30; i = i + 1) begin
      rst = (i == 17);
      @(posedge clk); #1;
      r = rst ? 3'd0 : (r == 3'd4 ? 3'd0 : r + 3'd1);
      n = n + 1; if (q !== r) bad = bad + 1;
    end
  """ + _END)

TASKS = _T
