module onehot4_decode(input logic [1:0] index, output logic [3:0] onehot);
  always_comb begin
    onehot = 4'b0001 << index;
  end
endmodule
