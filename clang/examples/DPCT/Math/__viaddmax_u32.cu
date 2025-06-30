// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const unsigned int a, const unsigned int b,
                     const unsigned int c) {
  // Start
  __viaddmax_u32(a /*const unsigned int*/, b /*const unsigned int*/,
                 c /*const unsigned int*/);
  // End
}
