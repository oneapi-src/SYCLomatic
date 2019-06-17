// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/device001.sycl.cpp

int main(int argc, char **argv) {

  // CHECK: syclct::sycl_device_info deviceProp;
  cudaDeviceProp deviceProp;

  // CHECK: if (deviceProp.get_mode() == syclct::compute_mode::prohibited) {
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    return 0;
  }

// CHECK:/*
// CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:int major = deviceProp.get_major_version();
  int major = deviceProp.major;
// CHECK:/*
// CHECK-NEXT:SYCLCT1006:{{[0-9]+}}: SYCL doesn't provide standard API to differentiate between integrated/discrete GPU devices. Consider to re-implement the code which depends on this field
// CHECK-NEXT:*/
// CHECK-NEXT:int integrated = deviceProp.get_integrated();
  int integrated = deviceProp.integrated;

// CHECK:/*
// CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:deviceProp.set_major_version(1);
  deviceProp.major=1;

// CHECK:/*
// CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:int minor = deviceProp.get_minor_version();
  int minor = deviceProp.minor;

// CHECK:/*
// CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:deviceProp.set_minor_version(120);
  deviceProp.minor=120;

  // CHECK:     char *name = deviceProp.get_name();
  char *name = deviceProp.name;

  // CHECK:     int clock = deviceProp.get_max_clock_frequency();
  int clock = deviceProp.clockRate;
  int xxxx = 10;
  int yyyy = 5;

  // CHECK:  deviceProp.set_max_clock_frequency ( xxxx * 100 + yyyy);
  deviceProp.clockRate = xxxx * 100 + yyyy;

  // CHECK:     int count = deviceProp.get_max_compute_units();
  int count = deviceProp.multiProcessorCount;

// CHECK:/*
// CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:int n = deviceProp.get_minor_version() / deviceProp.get_major_version();
  int n = deviceProp.minor / deviceProp.major;

  // CHECK: size_t memsize = deviceProp.get_global_mem_size();
  size_t memsize = deviceProp.totalGlobalMem;

  return 0;
}
