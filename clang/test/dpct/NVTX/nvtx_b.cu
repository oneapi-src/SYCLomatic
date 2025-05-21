// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-12.9
// RUN: dpct --format-range=none -in-root %S -out-root %T/nvtx_b %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/nvtx_b/nvtx_b.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
#include "nvToolsExtCuda.h"

int main() {
  CUdevice *device;
  cuDeviceGet(device, 0);
  // CHECK:     /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvtxNameCuDeviceA is not supported.
  // CHECK-NEXT: */
  nvtxNameCuDeviceA(*device, "nvtx_device");
}
