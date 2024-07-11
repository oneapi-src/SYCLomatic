// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(double d1, double d2) {
  // Start
  __dsub_rn(d1 /*double*/, d2 /*double*/);
  // End
}
