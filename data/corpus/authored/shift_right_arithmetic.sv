module shift_right_arithmetic(input logic signed [31:0] value, input logic [4:0] amount, output logic signed [31:0] result);
  assign result = value >>> amount;
endmodule
