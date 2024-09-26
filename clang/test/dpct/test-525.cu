// RUN: dpct --format-range=none -out-root %T/test-525 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/test-525/test-525.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/test-525/test-525.dp.cpp -o %T/test-525/test-525.dp.o %}
#include <cuda_runtime.h>
class C {
  int nDevices;
public:
  void problem() {
    // CHECK: nDevices = dpct::device_count();
    cudaGetDeviceCount(&nDevices);
  }
};

