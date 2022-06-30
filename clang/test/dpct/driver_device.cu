// RUN: dpct --format-range=none -out-root %T/driver_device %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_device/driver_device.dp.cpp
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM 1
#define MY_SAFE_CALL(CALL) do {    \
  int Error = CALL;                \
} while (0)
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

  // CHECK: MY_SAFE_CALL((device = 0, 0));
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

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuDeviceGetAttribute is not supported.
  // CHECK-NEXT: */
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);

  // CHECK: MY_SAFE_CALL((result1 = dpct::dev_mgr::instance().get_device(device).get_max_compute_units(), 0));
  MY_SAFE_CALL(cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

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

  // CHECK: MY_SAFE_CALL((result1 = dpct::dev_mgr::instance().device_count(), 0));
  MY_SAFE_CALL(cuDeviceGetCount(&result1));

  // CHECK: result1 = dpct::dev_mgr::instance().device_count();
  cuDeviceGetCount(&result1);

  // CHECK: MY_SAFE_CALL((result1 = dpct::dev_mgr::instance().device_count(), 0));
  MY_SAFE_CALL(cuDeviceGetCount(&result1));

  char name[100];

  // CHECK: memcpy(name, dpct::dev_mgr::instance().get_device(device).get_info<sycl::info::device::name>().c_str(), 90);
  cuDeviceGetName(name, 90, device);

  // CHECK: MY_SAFE_CALL((memcpy(name, dpct::dev_mgr::instance().get_device(device).get_info<sycl::info::device::name>().c_str(), 90), 0));
  MY_SAFE_CALL(cuDeviceGetName(name, 90, device));
  // CHECK: size = dpct::dev_mgr::instance().get_device(device).get_device_info().get_global_mem_size();
  cuDeviceTotalMem(&size, device);

  return 0;
}
