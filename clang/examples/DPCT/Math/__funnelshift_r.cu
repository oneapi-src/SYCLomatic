__global__ void test(unsigned int ui_a, unsigned int ui_b, unsigned int ui_c) {
  // Start
  __funnelshift_r(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
                  ui_c /*unsigned int*/);
  // End
}