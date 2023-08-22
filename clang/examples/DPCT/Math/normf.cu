// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(int i, const float *f) {
  // Start
  normf(i /*int*/, f /*const float **/);
  // End
}
