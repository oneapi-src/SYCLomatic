// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(float f1, float f2) {
  // Start
  __fmul_rd(f1 /*float*/, f2 /*float*/);
  // End
}
