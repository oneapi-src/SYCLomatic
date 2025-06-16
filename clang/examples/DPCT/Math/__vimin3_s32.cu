// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const int a, const int b, const int c) {
  // Start
  __vimin3_s32(a /*const int*/, b /*const int*/, c /*const int*/);
  // End
}
