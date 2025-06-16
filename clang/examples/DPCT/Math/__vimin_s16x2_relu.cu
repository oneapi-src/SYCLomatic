// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const unsigned int a, const unsigned int b) {
  // Start
  __vimin_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/);
  // End
}
