__global__ void test(int i_a, int i_b, int i_c, unsigned int ui_a,
                     unsigned int ui_b, unsigned int ui_c, uchar4 uc4_a,
                     uchar4 uc4_b, char4 c4_a, char4 c4_b) {
  // Start
  __dp4a(i_a /*int*/, i_b /*int*/, i_c /*int*/);
  __dp4a(ui_a /*unsigned int*/, ui_b /*unsigned int*/, ui_c /*unsigned int*/);
  __dp4a(uc4_a /*uchar4*/, uc4_b /*uchar4*/, ui_c /*unsigned int*/);
  __dp4a(c4_a /*char4*/, c4_b /*char4*/, i_c /*int*/);
  // End
}
