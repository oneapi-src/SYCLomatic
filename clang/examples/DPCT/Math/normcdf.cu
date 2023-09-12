// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(double d) {
  // Start
  normcdf(d /*double*/);
  // End
}
