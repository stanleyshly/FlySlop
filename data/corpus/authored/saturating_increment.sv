module saturating_increment(input logic [7:0] value, output logic [7:0] result);
  assign result = (&value) ? value : value + 8'd1;
endmodule
