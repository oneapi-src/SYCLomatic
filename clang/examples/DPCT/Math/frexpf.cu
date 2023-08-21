__global__ void test(float f, int *pi) {
  // Start
  frexpf(f /*float*/, pi /*int **/);
  // End
}
