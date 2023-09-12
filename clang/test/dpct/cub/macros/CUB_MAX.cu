// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/CUB_MAX %S/CUB_MAX.cu --cuda-include-path="%cuda-path/include" -- -std=c++17 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/CUB_MAX/CUB_MAX.dp.cpp %s

#include <cub/cub.cuh>

__global__ void f() {
  // CHECK: int a = std::max(1, 2);
  [[maybe_unused]] int a = CUB_MAX(1, 2);

  // CHECK: a = std::min(3, 4);
  a = CUB_MIN(3, 4);
}
