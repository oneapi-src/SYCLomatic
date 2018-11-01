// RUN: syclct -out-root %T %S/../.././test/./syclct/writing-files.cu -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/writing-files.sycl.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/test-header.sycl.hpp --match-full-lines %S/test-header.cuh

#include "test-header.cuh"

__global__ void foo() {
  // CHECK: size_t tix = item_{{[a-f0-9]+}}.get_local_id(0);
  // CHECK: size_t tiy = item_{{[a-f0-9]+}}.get_local_id(1);
  // CHECK: size_t tiz = item_{{[a-f0-9]+}}.get_local_id(2);

  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // size_t bix = blockIdx.x;
  // size_t biy = blockIdx.y;
  // size_t biz = blockIdx.z;

  // CHECK: size_t bdx = item_{{[a-f0-9]+}}.get_local_range().get(0);
  // CHECK: size_t bdy = item_{{[a-f0-9]+}}.get_local_range().get(1);
  // CHECK: size_t bdz = item_{{[a-f0-9]+}}.get_local_range().get(2);

  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // size_t gdx = gridDim.x;
  // size_t gdy = gridDim.y;
  // size_t gdz = gridDim.z;
}
