// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(double d) {
  // Start
  cyl_bessel_i0(d /*double*/);
  // End
}
