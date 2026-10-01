module mux2(input logic sel, input logic [7:0] a, b, output logic [7:0] y);
  assign y = sel ? b : a;
endmodule
