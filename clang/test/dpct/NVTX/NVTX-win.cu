// UNSUPPORTED: system-linux
// RUN: dpct --format-range=none -in-root %S -out-root %T %S/NVTX-win.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/NVTX-win.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
#include "nvToolsExtCuda.h"

int main(){
  CUdevice* device;
  cuDeviceGet(device,0);
  // CHECK:     /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvtxNameCuDeviceW is not supported.
  // CHECK-NEXT: */
  nvtxNameCuDeviceW(*device,"nvtx_device");
}
