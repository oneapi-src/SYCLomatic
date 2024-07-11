// RUN: dpct --format-range=none -out-root %T/cuda_pi %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda_pi/cuda_pi.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_pi/cuda_pi.dp.cpp -o %T/cuda_pi/cuda_pi.dp.o %}

#include <cuda.h>
#include <math_constants.h>

__global__ void test() {
  // CHECK: const auto pi = 3.141592654F;
  const auto pi = CUDART_PI_F;
}
