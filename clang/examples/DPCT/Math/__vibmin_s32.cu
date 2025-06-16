// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const int a, const int b, bool *const pred) {
  // Start
  __vibmin_s32(a /*const int*/, b /*const int*/, pred /*bool *const*/);
  // End
}
