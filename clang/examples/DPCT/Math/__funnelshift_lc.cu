__global__ void test(unsigned int ui_a, unsigned int ui_b, unsigned int ui_c) {
  // Start
  __funnelshift_lc(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
                   ui_c /*unsigned int*/);
  // End
}