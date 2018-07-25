// RUN: cp %s %t
// RUN: cu2sycl %t -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %t

int main(int argc, char **argv) {

  // CHECK: cu2sycl::sycl_device_info deviceProp;
  cudaDeviceProp deviceProp;

  // CHECK: if (deviceProp.compute_mode() == cu2sycl::cm_prohibited) {
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    return 0;
  }

  // CHECK:     int major = deviceProp.major_version();
  int major = deviceProp.major;

  // CHECK: deviceProp.major_version() = 1;
  deviceProp.major = 1;

  // CHECK:     int minor = deviceProp.minor_version();
  int minor = deviceProp.minor;

  // CHECK: deviceProp.minor_version() = 120;
  deviceProp.minor = 120;

  // CHECK:     char *name = deviceProp.name();
  char *name = deviceProp.name;

  // CHECK:     int clock = deviceProp.max_clock_frequency();
  int clock = deviceProp.clockRate;
  int xxxx = 10;
  int yyyy = 5;

  // CHECK:  deviceProp.max_clock_frequency() = xxxx * 100 + yyyy;
  deviceProp.clockRate = xxxx * 100 + yyyy;

  // CHECK:     int count = deviceProp.max_compute_units();
  int count = deviceProp.multiProcessorCount;

  // CHECK: int n = deviceProp.minor_version() / deviceProp.major_version();
  int n = deviceProp.minor / deviceProp.major;
  return 0;
}
