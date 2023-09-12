__global__ void test(float f, int i) {
  // Start
  ldexpf(f /*float*/, i /*int*/);
  // End
}
