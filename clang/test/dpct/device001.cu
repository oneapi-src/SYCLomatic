// RUN: dpct --format-range=none --usm-level=none -out-root %T/device001 %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device001/device001.dp.cpp

#include <algorithm>

int main(int argc, char **argv) {

  // CHECK: dpct::device_info deviceProp;
  cudaDeviceProp deviceProp;

  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (false) {
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    return 0;
  }

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int major = deviceProp.get_major_version();
  int major = deviceProp.major;
// CHECK:/*
// CHECK-NEXT:DPCT1006:{{[0-9]+}}: SYCL does not provide a standard API to differentiate between integrated and discrete GPU devices.
// CHECK-NEXT:*/
// CHECK-NEXT:int integrated = deviceProp.get_integrated();
  int integrated = deviceProp.integrated;

  // CHECK: int warpSize = deviceProp.get_max_sub_group_size();
  int warpSize = deviceProp.warpSize;

  // CHECK: int maxThreadsPerMultiProcessor = deviceProp.get_max_work_items_per_compute_unit();
  int maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:deviceProp.set_major_version(1);
  deviceProp.major=1;

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int minor = deviceProp.get_minor_version();
  int minor = deviceProp.minor;

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
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

  // CHECK: int freq = deviceProp.get_memory_clock_rate();
  int freq = deviceProp.memoryClockRate;

  // CHECK: int buswidth = deviceProp.get_memory_bus_width();
  int buswidth = deviceProp.memoryBusWidth;

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1022:{{[0-9]+}}: There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of the code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  int maxGridSizeX = deviceProp.get_max_nd_range_size<int *>()[0];
  int maxGridSizeX = deviceProp.maxGridSize[0];

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1022:{{[0-9]+}}: There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of the code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  int *maxGridSize_ptr = deviceProp.get_max_nd_range_size<int *>();
  int *maxGridSize_ptr = deviceProp.maxGridSize;

  // CHECK:/*
  // CHECK-NEXT:DPCT1019:{{[0-9]+}}: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to adjust the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:size_t share_mem_size = deviceProp.get_local_mem_size();
  size_t share_mem_size = deviceProp.sharedMemPerBlock;
  // CHECK:/*
  // CHECK-NEXT:DPCT1019:{{[0-9]+}}: local_mem_size in SYCL is not a complete equivalent of sharedMemPerMultiprocessor in CUDA. You may need to adjust the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:size_t share_multi_proc_mem_size = deviceProp.get_local_mem_size();
  size_t share_multi_proc_mem_size = deviceProp.sharedMemPerMultiprocessor;

  // CHECK: sycl::range<3> grid(1, 1, deviceProp.get_max_compute_units() * (deviceProp.get_max_work_items_per_compute_unit() / deviceProp.get_max_sub_group_size()));
  dim3 grid(deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize));

// CHECK:/*
// CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
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
  // CHECK:   dpct::get_default_queue().parallel_for<dpct_kernel_name<class foo_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, deviceProp.get_max_compute_units()) * sycl::range<3>(1, 1, deviceProp.get_max_work_group_size()), sycl::range<3>(1, 1, deviceProp.get_max_work_group_size())),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           foo_kernel();
  // CHECK-NEXT:         });
  foo_kernel<<<deviceProp.multiProcessorCount, deviceProp.maxThreadsPerBlock,  deviceProp.maxThreadsPerBlock>>>();
}

void test2() {
  cudaLimit limit;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceSetLimit was removed because SYCL currently does not support setting resource limits on devices.
  // CHECK-NEXT: */
  cudaDeviceSetLimit(limit, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaDeviceSetLimit was replaced with 0 because SYCL currently does not support setting resource limits on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: int i = 0;
  int i = cudaDeviceSetLimit(limit, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaThreadSetLimit was removed because SYCL currently does not support setting resource limits on devices.
  // CHECK-NEXT: */
  cudaThreadSetLimit(limit, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaThreadSetLimit was replaced with 0 because SYCL currently does not support setting resource limits on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: int j = 0;
  int j = cudaThreadSetLimit(limit, 0);
}

void test3() {
  //CHECK:dpct::device_info deviceProp;
  cudaDeviceProp deviceProp;
  //CHECK:int a1 = deviceProp.get_max_work_item_sizes<int *>()[2];
  int a1 = deviceProp.maxThreadsDim[2];
  //CHECK:int *a1_ptr = deviceProp.get_max_work_item_sizes<int *>();
  int *a1_ptr = deviceProp.maxThreadsDim;
  //CHECK:/*
  //CHECK-NEXT:DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with memPitch. It was migrated to INT_MAX. You may need to rewrite the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a2 = INT_MAX;
  int a2 = deviceProp.memPitch;
  //CHECK:/*
  //CHECK-NEXT:DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with totalConstMem. It was migrated to 0. You may need to rewrite the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a3 = 0;
  int a3 = deviceProp.totalConstMem;
  //CHECK:/*
  //CHECK-NEXT:DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with regsPerBlock. It was not migrated. You need to rewrite the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a4 = deviceProp.regsPerBlock;
  int a4 = deviceProp.regsPerBlock;
  //CHECK:/*
  //CHECK-NEXT:DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with textureAlignment. It was migrated to dpct::get_current_device().get_info<sycl::info::device::mem_base_addr_align>(). You may need to rewrite the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a5 = dpct::get_current_device().get_info<sycl::info::device::mem_base_addr_align>();
  int a5 = deviceProp.textureAlignment;
  //CHECK:/*
  //CHECK-NEXT:DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with kernelExecTimeoutEnabled. It was migrated to false. You may need to rewrite the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a6 = false;
  int a6 = deviceProp.kernelExecTimeoutEnabled;
  //CHECK:int a7 = dpct::get_current_device().get_info<sycl::info::device::error_correction_support>();
  int a7 = deviceProp.ECCEnabled;
}

//CHECK:dpct::device_info* getcudaDevicePropPtr() {
//CHECK-NEXT:  return (dpct::device_info*)0;
//CHECK-NEXT:}
cudaDeviceProp* getcudaDevicePropPtr() {
  return (cudaDeviceProp*)0;
}

void test4() {
  //CHECK:int a1 = getcudaDevicePropPtr()->get_max_compute_units();
  int a1 = getcudaDevicePropPtr()->multiProcessorCount;

  //CHECK: a1 = getcudaDevicePropPtr()->get_memory_clock_rate();
  a1 = getcudaDevicePropPtr()->memoryClockRate;

  //CHECK: a1 = getcudaDevicePropPtr()->get_memory_bus_width();
  a1 = getcudaDevicePropPtr()->memoryBusWidth;
}

__device__ void test5() {
  //CHECK:dpct::device_info *pDeviceProp = nullptr;
  cudaDeviceProp *pDeviceProp = nullptr;
  [=]() {
    int a;
    //CHECK:a = sycl::min((int)pDeviceProp->get_global_mem_size(), 1000);
    a = std::min((int)pDeviceProp->totalGlobalMem, 1000);
    // CHECK:/*
    // CHECK-NEXT:DPCT1006:{{[0-9]+}}: SYCL does not provide a standard API to differentiate between integrated and discrete GPU devices.
    // CHECK-NEXT:*/
    // CHECK-NEXT: a = sycl::min(pDeviceProp->get_integrated(), 1000);
    a = std::min(pDeviceProp->integrated, 1000);
    // CHECK:/*
    // CHECK-NEXT:DPCT1019:{{[0-9]+}}: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to adjust the code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:a = sycl::min((int)pDeviceProp->get_local_mem_size(), 1000);
    a = std::min((int)pDeviceProp->sharedMemPerBlock, 1000);
    // CHECK:/*
    // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT: a = sycl::min(pDeviceProp->get_major_version(), 1000);
    a = std::min(pDeviceProp->major, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_max_clock_frequency(), 1000);
    a = std::min(pDeviceProp->clockRate, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_max_compute_units(), 1000);
    a = std::min(pDeviceProp->multiProcessorCount, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_memory_clock_rate(), 1000);
    a = std::min(pDeviceProp->memoryClockRate, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_memory_bus_width(), 1000);
    a = std::min(pDeviceProp->memoryBusWidth, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_max_sub_group_size(), 1000);
    a = std::min(pDeviceProp->warpSize, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_max_work_group_size(), 1000);
    a = std::min(pDeviceProp->maxThreadsPerBlock, 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_max_work_items_per_compute_unit(), 1000);
    a = std::min(pDeviceProp->maxThreadsPerMultiProcessor, 1000);
    // CHECK:/*
    // CHECK-NEXT:DPCT1022:{{[0-9]+}}: There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of the code.
    // CHECK-NEXT:*/
    // CHECK-NEXT: a = sycl::min(pDeviceProp->get_max_nd_range_size<int *>()[0], 1000);
    a = std::min(pDeviceProp->maxGridSize[0], 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_max_work_item_sizes<int *>()[0], 1000);
    a = std::min(pDeviceProp->maxThreadsDim[0], 1000);
    //CHECK:a = sycl::min(pDeviceProp->get_name()[0], 'A');
    a = std::min(pDeviceProp->name[0], 'A');
  }();
}
