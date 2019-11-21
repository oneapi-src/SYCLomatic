// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device001.dp.cpp

int main(int argc, char **argv) {

  // CHECK: dpct::device_info deviceProp;
  cudaDeviceProp deviceProp;

  // CHECK: if (deviceProp.get_mode() == dpct::compute_mode::prohibited) {
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    return 0;
  }

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int major = deviceProp.get_major_version();
  int major = deviceProp.major;
// CHECK:/*
// CHECK-NEXT:DPCT1006:{{[0-9]+}}: DPC++ does not provide a standard API to differentiate between integrated/ discrete GPU devices.
// CHECK-NEXT:*/
// CHECK-NEXT:int integrated = deviceProp.get_integrated();
  int integrated = deviceProp.integrated;

  // CHECK: int warpSize = deviceProp.get_max_sub_group_size();
  int warpSize = deviceProp.warpSize;

  // CHECK: int maxThreadsPerMultiProcessor = deviceProp.get_max_work_items_per_compute_unit();
  int maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:deviceProp.set_major_version(1);
  deviceProp.major=1;

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int minor = deviceProp.get_minor_version();
  int minor = deviceProp.minor;

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
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

  // CHECK: int count = deviceProp.get_max_compute_units();
  int count = deviceProp.multiProcessorCount;

  // CHECK: count = deviceProp.get_max_work_group_size();
  count = deviceProp.maxThreadsPerBlock;

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1022:{{[0-9]+}}: There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of the code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  int *maxGridSize = deviceProp.get_max_nd_range_size();
  int *maxGridSize = deviceProp.maxGridSize;

  // CHECK:/*
  // CHECK-NEXT:DPCT1019:{{[0-9]+}}: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to adjust the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:size_t share_mem_size = deviceProp.get_local_mem_size();
  size_t share_mem_size = deviceProp.sharedMemPerBlock;

  // CHECK: cl::sycl::range<3> grid(deviceProp.get_max_compute_units() * (deviceProp.get_max_work_items_per_compute_unit() / deviceProp.get_max_sub_group_size()), 1, 1);
  dim3 grid(deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize));

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int n = deviceProp.get_minor_version() / deviceProp.get_major_version();
  int n = deviceProp.minor / deviceProp.major;

  // CHECK: size_t memsize = deviceProp.get_global_mem_size();
  size_t memsize = deviceProp.totalGlobalMem;

  // CHECK: int i=true;
  int i=deviceProp.deviceOverlap;
  // CHECK: if(true){
  if(deviceProp.deviceOverlap){
  //dosomething.
  }
  return 0;
}

__global__ void foo_kernel(void)
{
}

void test()
{
  // CHECK: dpct::device_info deviceProp;
  cudaDeviceProp deviceProp;
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(deviceProp.get_max_compute_units(), 1, 1) * cl::sycl::range<3>(deviceProp.get_max_work_group_size(), 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(deviceProp.get_max_work_group_size(), 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class foo_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           foo_kernel();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  foo_kernel<<<deviceProp.multiProcessorCount, deviceProp.maxThreadsPerBlock,  deviceProp.maxThreadsPerBlock>>>();
}

void test2() {
  cudaLimit limit;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceSetLimit was removed. DPC++ currently doesn't support setting resource limits on devices.
  // CHECK-NEXT: */
  cudaDeviceSetLimit(limit, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaDeviceSetLimit was replaced with 0. DPC++ currently doesn't support setting resource limits on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: int i = 0;
  int i = cudaDeviceSetLimit(limit, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaThreadSetLimit was removed. DPC++ currently doesn't support setting resource limits on devices.
  // CHECK-NEXT: */
  cudaThreadSetLimit(limit, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaThreadSetLimit was replaced with 0. DPC++ currently doesn't support setting resource limits on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: int j = 0;
  int j = cudaThreadSetLimit(limit, 0);
}
