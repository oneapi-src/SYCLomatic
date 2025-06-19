// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const unsigned int a, const unsigned int b,
                     bool *const pred) {
  // Start
  __vibmin_u32(a /*const unsigned int*/, b /*const unsigned int*/,
               pred /*bool *const*/);
  // End
}
