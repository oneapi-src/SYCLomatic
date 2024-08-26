// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/warplevel/shuffle_without_exp %S/shuffle_without_exp.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warplevel/shuffle_without_exp/shuffle_without_exp.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void ShuffleDownKernel(int *data) {
  int tid = cub::LaneId();
  unsigned mask = 0x8;
  int val = tid;
  // CHECK: DPCT1007:{{.*}}: Migration of cub::ShuffleDown is not supported.
  data[tid] = cub::ShuffleDown<8>(val, 3, 6, mask);
}

__global__ void ShuffleUpKernel(int *data) {
  int tid = cub::LaneId();
  unsigned mask = 0x8;
  int val = tid;
  // CHECK: DPCT1007:{{.*}}: Migration of cub::ShuffleUp is not supported.
  data[tid] = cub::ShuffleUp<8>(val, 3, 6, mask);
}
