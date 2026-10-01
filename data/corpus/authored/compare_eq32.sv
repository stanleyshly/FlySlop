module compare_eq32(input logic [31:0] a, b, output logic equal);
  assign equal = (a == b);
endmodule
