__global__ void test(double d, int *pi) {
  // Start
  frexp(d /*double*/, pi /*int **/);
  // End
}
