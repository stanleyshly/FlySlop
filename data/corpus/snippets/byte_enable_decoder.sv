module wben_decoder
(
  input  logic [1:0]  offset,    // 2-bit word offset within 128-bit cacheline
  output logic [15:0] wben       // 16-bit byte enable output
);

  always_comb begin
    // Enable 4 consecutive bytes based on word offset
    case (offset)
      2'b00: wben = 16'b0000_0000_0000_1111; // Bytes 0-3 (offset 0x0)
      2'b01: wben = 16'b0000_0000_1111_0000; // Bytes 4-7 (offset 0x4)
      2'b10: wben = 16'b0000_1111_0000_0000; // Bytes 8-11 (offset 0x8)
      2'b11: wben = 16'b1111_0000_0000_0000; // Bytes 12-15 (offset 0xC)
      default: wben = 16'b0000_0000_0000_0000;
    endcase
  end

endmodule