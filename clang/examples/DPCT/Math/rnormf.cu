// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(int i, const float *f) {
  // Start
  rnormf(i /*int*/, f /*const float **/);
  // End
}
