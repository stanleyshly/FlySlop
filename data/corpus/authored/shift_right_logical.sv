module shift_right_logical(input logic [31:0] value, input logic [4:0] amount, output logic [31:0] result);
  assign result = value >> amount;
endmodule
