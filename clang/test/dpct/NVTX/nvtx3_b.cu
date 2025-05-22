// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/nvtx3_b %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/nvtx3_b/nvtx3_b.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
#include "nvtx3/nvToolsExtCuda.h"

int main() {
  CUdevice *device;
  cuDeviceGet(device, 0);
  // CHECK:     /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvtxNameCuDeviceA is not supported.
  // CHECK-NEXT: */
  nvtxNameCuDeviceA(*device, "nvtx_device");
}
