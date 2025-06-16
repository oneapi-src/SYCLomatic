__global__ void test(int i_a, int i_b, int i_c, unsigned int ui_a,
                     unsigned int ui_b, unsigned int ui_c, ushort2 us2_a,
                     uchar4 uc4_b, short2 s2_a, char4 c4_b) {
  // Start
  __dp2a_lo(i_a /*int*/, i_b /*int*/, i_c /*int*/);
  __dp2a_lo(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
            ui_c /*unsigned int*/);
  __dp2a_lo(us2_a /*ushort2*/, uc4_b /*uchar4*/, ui_c /*unsigned int*/);
  __dp2a_lo(s2_a /*short2*/, c4_b /*char4*/, i_c /*int*/);
  // End
}
