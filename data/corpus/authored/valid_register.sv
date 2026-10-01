module valid_register(input logic clk, reset, set_valid, clear_valid, output logic valid);
  always_ff @(posedge clk) begin
    if (reset || clear_valid) valid <= 1'b0;
    else if (set_valid) valid <= 1'b1;
  end
endmodule
