// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/replace-cuda-header-includes.sycl.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include "test-header.cuh"
// CHECK: // First function
#include "test-header.cuh"
#include <cuda.h>
#include <device_functions.h>
#include <cufft.h>
// First function
__global__ void foo() {
  // CHECK: size_t tix = item.get_local_id(0);
  // CHECK: size_t tiy = item.get_local_id(1);
  // CHECK: size_t tiz = item.get_local_id(2);

  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // size_t bix = blockIdx.x;
  // size_t biy = blockIdx.y;
  // size_t biz = blockIdx.z;

  // CHECK: size_t bdx = item.get_local_range().get(0);
  // CHECK: size_t bdy = item.get_local_range().get(1);
  // CHECK: size_t bdz = item.get_local_range().get(2);

  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // size_t gdx = gridDim.x;
  // size_t gdy = gridDim.y;
  // size_t gdz = gridDim.z;
}
