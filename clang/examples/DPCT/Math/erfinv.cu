// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(double d) {
  // Start
  erfinv(d /*double*/);
  // End
}
