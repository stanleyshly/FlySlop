module pc_select(input logic [31:0] pc, target, input logic take_branch, output logic [31:0] next_pc);
  assign next_pc = take_branch ? target : pc + 32'd4;
endmodule
