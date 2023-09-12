// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/sync_stream %S/sync_stream.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/sync_stream/sync_stream.dp.cpp --match-full-lines %s

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdio.h>

__global__ void kernel(int *res) {
  // CHECK: q_ct1.wait();
  cub::SyncStream(0);
}

int main() {
  cudaStream_t s;
  cudaStreamCreate(&s);

  // CHECK: s->wait();
  cub::SyncStream(s);

  // CHECK: q_ct1.wait();
  cub::SyncStream(0);

  // CHECK: q_ct1.wait();
  cub::SyncStream((cudaStream_t)(uintptr_t)1);

  // CHECK: q_ct1.wait();
  cub::SyncStream((cudaStream_t)(uintptr_t)2);

  cudaStreamDestroy(s);
  return 0;
}
