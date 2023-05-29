// RUN: dpct --format-range=none -out-root %T/driver_device %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_device/driver_device.dp.cpp
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define NUM 1
#define MY_SAFE_CALL(CALL) do {    \
  int Error = CALL;                \
} while (0)

void test() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuInit was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, 0);

  int result0, result1, result2, result3, result4, result5;
  // CHECK: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support a device property functionally compatible with CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY. It was migrated to get_global_mem_size. You may need to adjust the value of get_global_mem_size for the specific device.
  // CHECK-NEXT: */
  // CHECK: result0 = dpct::dev_mgr::instance().get_device(device).get_global_mem_size();
  cuDeviceGetAttribute(&result0, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
  std::cout << " result0 " << result0 << std::endl;
  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_max_sub_group_size();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
  std::cout << " result1 " << result1 << std::endl;
  // CHECK: result2 = dpct::dev_mgr::instance().get_device(device).get_max_work_group_size();
  cuDeviceGetAttribute(&result2,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
  std::cout << " result2 " << result2 << std::endl;
  // CHECK: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support a device property functionally compatible with CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. It was migrated to get_mem_base_addr_align. You may need to adjust the value of get_mem_base_addr_align for the specific device.
  // CHECK-NEXT: */
  // CHECK: result3 = dpct::dev_mgr::instance().get_device(device).get_mem_base_addr_align();
  cuDeviceGetAttribute(&result3,CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device);
  std::cout << " result3 " << result3 << std::endl;
  // CHECK: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support a device property functionally compatible with CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK. It was migrated to get_max_register_size_per_work_group. You may need to adjust the value of get_max_register_size_per_work_group for the specific device.
  // CHECK-NEXT: */
  // CHECK: result4 = dpct::dev_mgr::instance().get_device(device).get_max_register_size_per_work_group();
  cuDeviceGetAttribute(&result4,CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
  std::cout << " result4 " << result4 << std::endl;
  // CHECK: result5 = dpct::dev_mgr::instance().get_device(device).has(sycl::aspect::usm_host_allocations)();
  cuDeviceGetAttribute(&result5,CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device);
  std::cout << " result5 " << result5 << std::endl;
}

int main(){
  int result1, result2;

  int *presult1 = &result1, *presult2 = &result2;
  size_t size;
  // CHECK: int device;
  CUdevice device;

  // CHECK: int *pdevice = &device;
  CUdevice *pdevice = &device;

  // CHECK: device = 0;
  cuDeviceGet(&device, 0);

  // CHECK: device = NUM;
  cuDeviceGet(&device, NUM);

  // CHECK: *pdevice = 0;
  cuDeviceGet(pdevice, 0);

  // CHECK: *((int *)pdevice) = 0;
  cuDeviceGet((CUdevice *)pdevice, 0);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(device = 0));
  MY_SAFE_CALL(cuDeviceGet(&device, 0));

  // CHECK: /*
  // CHECK-NEXT: DPCT1076:{{[0-9]+}}: The device attribute was not recognized. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuDeviceGetAttribute(&result1, attr, device);
  CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
  cuDeviceGetAttribute(&result1, attr, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_major_version();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_minor_version();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_integrated();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_max_clock_frequency();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_max_compute_units();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).is_native_atomic_supported();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_device_info().get_max_work_item_sizes().get(0);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_device_info().get_local_mem_size();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);

  // CHECK: int context;
  CUcontext context;
  // CHECK: unsigned int flags = 0;
  unsigned int flags = CU_CTX_MAP_HOST;
  // CHECK: flags += 0;
  flags += CU_CTX_SCHED_BLOCKING_SYNC;
  // CHECK: flags += 0;
  flags += CU_CTX_SCHED_SPIN;
  if (cuCtxCreate(&context, flags, device) == CUDA_SUCCESS) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxSetCacheConfig was removed because SYCL currently does not support setting cache config on devices.
  // CHECK-NEXT: */
  cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxSetLimit was removed because SYCL currently does not support setting resource limits on devices.
  // CHECK-NEXT: */
  cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 10);
  size_t printfsize;

  // CHECK: printfsize = INT_MAX;
  cuCtxGetLimit(&printfsize, CU_LIMIT_PRINTF_FIFO_SIZE);


  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_max_work_group_size();
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(result1 = dpct::dev_mgr::instance().get_device(device).get_max_compute_units()));
  MY_SAFE_CALL(cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1028:{{[0-9]+}}: The cuDeviceGetAttribute was not migrated because parameter CU_DEVICE_ATTRIBUTE_GPU_OVERLAP is unsupported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device);

  // CHECK: result1 = dpct::dev_mgr::instance().get_device(device).get_major_version();
  // CHECK: result2 = dpct::dev_mgr::instance().get_device(device).get_minor_version();
  cuDeviceComputeCapability(&result1, &result2, device);

  // CHECK: MY_SAFE_CALL([&](){
  // CHECK-NEXT:   result1 = dpct::dev_mgr::instance().get_device(device).get_major_version();
  // CHECK-NEXT:   result2 = dpct::dev_mgr::instance().get_device(device).get_minor_version();
  // CHECK-NEXT:   return 0;
  // CHECK-NEXT: }());
  MY_SAFE_CALL(cuDeviceComputeCapability(&result1, &result2, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
  // CHECK-NEXT: */
  // CHECK-NEXT: result1 = std::stoi(dpct::get_current_device().get_info<sycl::info::device::version>());
  cuDriverGetVersion(&result1);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(result1 = dpct::dev_mgr::instance().device_count()));
  MY_SAFE_CALL(cuDeviceGetCount(&result1));

  // CHECK: result1 = dpct::dev_mgr::instance().device_count();
  cuDeviceGetCount(&result1);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(result1 = dpct::dev_mgr::instance().device_count()));
  MY_SAFE_CALL(cuDeviceGetCount(&result1));

  char name[100];

  // CHECK: memcpy(name, dpct::dev_mgr::instance().get_device(device).get_info<sycl::info::device::name>().c_str(), 90);
  cuDeviceGetName(name, 90, device);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(memcpy(name, dpct::dev_mgr::instance().get_device(device).get_info<sycl::info::device::name>().c_str(), 90)));
  MY_SAFE_CALL(cuDeviceGetName(name, 90, device));
  // CHECK: size = dpct::dev_mgr::instance().get_device(device).get_device_info().get_global_mem_size();
  cuDeviceTotalMem(&size, device);

  return 0;
}
