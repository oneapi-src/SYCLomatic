// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(unsigned int u) {
  // Start
  __vneg2(u /*unsigned int*/);
  // End
}
