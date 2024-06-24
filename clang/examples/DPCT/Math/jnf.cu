// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(int i, float f) {
  // Start
  jnf(i /*int*/, f /*float*/);
  // End
}
