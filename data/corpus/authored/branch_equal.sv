module branch_equal(input logic [31:0] a, b, input logic branch_en, output logic take_branch);
  assign take_branch = branch_en && (a == b);
endmodule
