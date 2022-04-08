// RUN: c2s --format-range=none -out-root %T/builtin_warpSize %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/builtin_warpSize/builtin_warpSize.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__ void foo(){
  // CHECK: int a = item_ct1.get_sub_group().get_local_range().get(0);
  // CHECK-NEXT: int warpSize = 1;
  // CHECK-NEXT: warpSize = 2;
  // CHECK-NEXT: int c= warpSize;
  int a = warpSize;
  int warpSize = 1;
  warpSize = 2;
  int c= warpSize;
}

// CHECK: void bar(sycl::nd_item<3> item_ct1){
// CHECK-NEXT:   int a = sycl::max((int)item_ct1.get_sub_group().get_local_range().get(0), 0);
// CHECK-NEXT:   int warpSize = 1;
// CHECK-NEXT:   int b = sycl::max(warpSize, 0);
// CHECK-NEXT: }
__global__ void bar(){
  int a = max(warpSize, 0);
  int warpSize = 1;
  int b = max(warpSize, 0);
}

// CHECK: int tensorPos(const int ct, sycl::nd_item<3> item_ct1, int numLane = 0) {
// CHECK-NEXT:   if (!numLane) numLane = item_ct1.get_sub_group().get_local_range().get(0);
// CHECK-NEXT:   int r = ct * numLane;
// CHECK-NEXT:   return r;
// CHECK-NEXT: }
__device__ int tensorPos(const int ct, const int numLane = warpSize) {
  int r = ct * numLane;
  return r;
}

// CHECK: int tensorPos(const int ct, sycl::nd_item<3> item_ct1, int numLane);
__device__ int tensorPos(const int ct, const int numLane);





// CHECK: int tensorPos2(const int ct, sycl::nd_item<3> item_ct1, int numLane);
__device__ int tensorPos2(const int ct, const int numLane);

// CHECK: int tensorPos2(const int ct, sycl::nd_item<3> item_ct1, int numLane) {
// CHECK-NEXT:   if (!numLane) numLane = item_ct1.get_sub_group().get_local_range().get(0);
// CHECK-NEXT:   int r = ct * numLane;
// CHECK-NEXT:   return r;
// CHECK-NEXT: }
__device__ int tensorPos2(const int ct, const int numLane) {
  int r = ct * numLane;
  return r;
}

// CHECK: int tensorPos2(const int ct, sycl::nd_item<3> item_ct1, int numLane = 0);
__device__ int tensorPos2(const int ct, const int numLane = warpSize);


// CHECK: int tensorPos3(const int ct, sycl::nd_item<3> item_ct1, int numLane = 0) {}
__device__ int tensorPos3(const int ct, const int numLane = warpSize) {}