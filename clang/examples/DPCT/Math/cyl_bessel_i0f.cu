// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(float f) {
  // Start
  cyl_bessel_i0f(f /*float*/);
  // End
}
