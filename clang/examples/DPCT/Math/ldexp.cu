__global__ void test(double d, int i) {
  // Start
  ldexp(d /*double*/, i /*int*/);
  // End
}
