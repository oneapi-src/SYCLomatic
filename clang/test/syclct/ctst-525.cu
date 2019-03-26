// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/ctst-525.sycl.cpp --match-full-lines %s
#include <cuda_runtime.h>
class C {
  int nDevices;
public:
  void problem() {
    // CHECK: nDevices = syclct::get_device_manager().device_count();
    cudaGetDeviceCount(&nDevices);
  }
};
