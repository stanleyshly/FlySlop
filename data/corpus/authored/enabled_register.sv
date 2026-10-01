module enabled_register(input logic clk, reset, en, input logic [7:0] d, output logic [7:0] q);
  always_ff @(posedge clk) begin
    if (reset) q <= 8'b0;
    else if (en) q <= d;
  end
endmodule
