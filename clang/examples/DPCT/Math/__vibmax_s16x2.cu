// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const unsigned int a, const unsigned int b,
                     bool *const pred_hi, bool *const pred_lo) {
  // Start
  __vibmax_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
                 pred_hi /*bool *const*/, pred_lo /*bool *const*/);
  // End
}
