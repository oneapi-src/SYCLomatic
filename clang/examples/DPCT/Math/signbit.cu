__global__ void test(float f, double d) {
  // Start
  signbit(f /*float*/);
  signbit(d /*double*/);
  // End
}
