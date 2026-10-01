module counter4(input logic clk, reset, en, output logic [3:0] count);
  always_ff @(posedge clk) begin
    if (reset) count <= 4'b0;
    else if (en) count <= count + 1'b1;
  end
endmodule
