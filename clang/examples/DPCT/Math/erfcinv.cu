// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(double d) {
  // Start
  erfcinv(d /*double*/);
  // End
}
