// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const int a, const int b, const int c) {
  // Start
  __vimax3_s32_relu(a /*const int*/, b /*const int*/, c /*const int*/);
  // End
}
