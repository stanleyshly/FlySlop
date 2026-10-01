module compare_signed_lt(input logic signed [31:0] a, b, output logic less);
  assign less = (a < b);
endmodule
