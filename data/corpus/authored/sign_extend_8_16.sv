module sign_extend_8_16(input logic [7:0] value, output logic [15:0] result);
  assign result = {{8{value[7]}}, value};
endmodule
