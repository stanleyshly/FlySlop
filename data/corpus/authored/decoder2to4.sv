module decoder2to4(input logic en, input logic [1:0] sel, output logic [3:0] y);
  always_comb begin
    y = 4'b0000;
    if (en) y[sel] = 1'b1;
  end
endmodule
