// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(float f) {
  // Start
  erfcinvf(f /*float*/);
  // End
}
