// RUN: dpct -out-root %T %s -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/ctst-525.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
class C {
  int nDevices;
public:
  void problem() {
    // CHECK: nDevices = dpct::get_device_manager().device_count();
    cudaGetDeviceCount(&nDevices);
  }
};
