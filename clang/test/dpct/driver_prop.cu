// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/driver_prop %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_prop/driver_prop.dp.cpp


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

int main(){
  // CHECK: int device;
  CUdevice device;
  // CHECK: std::array<unsigned char, 16> uid;
  CUuuid uid;
  // CHECK: auto res = DPCT_CHECK_ERROR(uid = dpct::get_device(device).get_device_info().get_uuid());
  auto res = cuDeviceGetUuid(&uid, device);
  // CHECK: uid = dpct::get_device(device).get_device_info().get_uuid();
  cuDeviceGetUuid(&uid, device);
  return 0;
}
