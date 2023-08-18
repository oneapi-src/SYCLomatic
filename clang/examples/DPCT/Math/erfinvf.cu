// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(float f) {
  // Start
  erfinvf(f /*float*/);
  // End
}
