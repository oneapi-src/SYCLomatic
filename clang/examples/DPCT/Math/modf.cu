__global__ void test(double d, double *pd) {
  // Start
  modf(d /*double*/, pd /*double **/);
  // End
}
