module shift_left2(input logic [31:0] value, output logic [31:0] result);
  assign result = {value[29:0], 2'b00};
endmodule
