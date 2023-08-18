// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(int i, const double *d) {
  // Start
  norm(i /*int*/, d /*const double **/);
  // End
}
