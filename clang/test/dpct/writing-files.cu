// RUN: c2s --format-range=none -out-root %T/writing-files %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/writing-files/writing-files.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/writing-files/test-header.dp.hpp --match-full-lines %S/test-header.cuh

#include "test-header.cuh"

__global__ void foo() {
  // CHECK: size_t tix = item_ct1.get_local_id(2);
  // CHECK: size_t tiy = item_ct1.get_local_id(1);
  // CHECK: size_t tiz = item_ct1.get_local_id(0);

  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // size_t bix = blockIdx.x;
  // size_t biy = blockIdx.y;
  // size_t biz = blockIdx.z;

  // CHECK: size_t bdx = item_ct1.get_local_range(2);
  // CHECK: size_t bdy = item_ct1.get_local_range(1);
  // CHECK: size_t bdz = item_ct1.get_local_range(0);

  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // size_t gdx = gridDim.x;
  // size_t gdy = gridDim.y;
  // size_t gdz = gridDim.z;
}
