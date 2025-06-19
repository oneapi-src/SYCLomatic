// Option: --use-dpcpp-extensions=intel_device_math

__global__ void test(const int a, const int b) {
  // Start
  __vimin_s32_relu(a /*const int*/, b /*const int*/);
  // End
}
