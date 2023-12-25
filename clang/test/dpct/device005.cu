// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --no-dpcpp-extensions=device_info -out-root %T/device005 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device005/device005.dp.cpp

#include <cuda.h>
#include <iostream>

int main() {
  CUuuid uid;
  CUdevice device;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with cuDeviceGetUuid. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuDeviceGetUuid(&uid, device);
  cuDeviceGetUuid(&uid, device);
  return 0;
}
