// UNSUPPORTED: cuda-8.0
// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/thrust-header.sycl.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK-NEXT: #include <algorithm>
// CHECK: #include <syclct/syclct_thrust.hpp>
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
int main() {

  int *mapsp1D, *mapspkeyD,*mapspvalD;
  int numsH=10;

  cudaMalloc(&mapsp1D, numsH*sizeof(int));
  cudaMalloc(&mapspkeyD, numsH*sizeof(int));
  cudaMalloc(&mapspvalD, numsH*sizeof(int));

//  thrust::device_ptr<int> mapsp1T(mapsp1D);
//  thrust::device_ptr<int> mapspkeyT(mapspkeyD);
//  thrust::device_ptr<int> mapspvalT(mapspvalD);

//  thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
//  thrust::sequence(mapspvalT, mapspvalT + numsH);
//  thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);

}
