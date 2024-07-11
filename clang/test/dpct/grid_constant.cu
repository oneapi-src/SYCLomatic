// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6
// RUN: dpct --format-range=none -out-root %T/grid_constant %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/grid_constant/grid_constant.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/grid_constant/grid_constant.dp.cpp -o %T/grid_constant/grid_constant.dp.o %}

#include <cuda.h>

// CHECK: void kernel(const float x) {}
__global__ void kernel(const __grid_constant__ float x) {}
