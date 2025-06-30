// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(float f0, float f1, float f2) {
  // Start
  __fmaf_ieee_rd(f0 /*float*/, f1 /*float*/, f2 /*float*/);
  // End
}
