module byte_mask(input logic [3:0] enables, output logic [31:0] mask);
  assign mask = {{8{enables[3]}}, {8{enables[2]}}, {8{enables[1]}}, {8{enables[0]}}};
endmodule
