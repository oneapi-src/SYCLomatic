// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
//  RUN: dpct --assume-nd-range-dim=1 --format-range=none -out-root %T/with_this_nd_item %s --cuda-include-path="%cuda-path/include" -use-experimental-features=free-function-queries -- -x cuda --cuda-host-only -fno-delayed-template-parsing -std=c++14
//  RUN: FileCheck --input-file %T/with_this_nd_item/with_this_nd_item.dp.cpp --match-full-lines %s

#include "cooperative_groups.h"

// CHECK: #define TB(b) auto b = sycl::ext::oneapi::experimental::this_group<1>();
#define TB(b) cg::thread_block b = cg::this_thread_block();
// CHECK:/*
// CHECK-NEXT:DPCT1088:{{[0-9]+}}: The macro definition has multiple migration results in the dimension of free queries function that could not be unified. You may need to modify the code.
// CHECK-NEXT:*/
// CHECK-NEXT: #define TB1(b) auto b = sycl::ext::oneapi::experimental::this_group<dpct_placeholder /* Fix the dimension manually */>();
#define TB1(b) cg::thread_block b = cg::this_thread_block();

namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: void test1() {
// CHECK-NEXT:  auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<1>();
// CHECK-NEXT:  int a = item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0) + item_ct1.get_group(0) +
// CHECK-NEXT:  item_ct1.get_local_range(0) + item_ct1.get_local_id(0);
// CHECK-NEXT:  sycl::group<1> cta = sycl::ext::oneapi::experimental::this_group<1>();
// CHECK-NEXT:  sycl::group<1> b0 = sycl::ext::oneapi::experimental::this_group<1>(), b1 = sycl::ext::oneapi::experimental::this_group<1>();
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT:  */
// CHECK-NEXT:  item_ct1.barrier();
// CHECK-NEXT:  TB(b);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT:  */
// CHECK-NEXT:  a = (item_ct1.barrier(), sycl::all_of_group(sycl::ext::oneapi::experimental::this_group<1>(), a));
// CHECK-NEXT:  sycl::all_of_group(sycl::ext::oneapi::experimental::this_sub_group(), a);
// CHECK-NEXT:  dpct::select_from_sub_group(sycl::ext::oneapi::experimental::this_sub_group(), a, a);
// CHECK-NEXT:}
__global__ void test1() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();
  cg::sync(cta);
  TB(b);
  a = __syncthreads_and(a);
  __all(a);
  __shfl(a, a);
}

// CHECK: void test2() {
// CHECK-NEXT:  auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
// CHECK-NEXT:  int a = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2) + item_ct1.get_group(2) +
// CHECK-NEXT:  item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
// CHECK-NEXT:  sycl::group<3> cta = sycl::ext::oneapi::experimental::this_group<3>();
// CHECK-NEXT:  sycl::group<3> b0 = sycl::ext::oneapi::experimental::this_group<3>(), b1 = sycl::ext::oneapi::experimental::this_group<3>();
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT:  */
// CHECK-NEXT:  item_ct1.barrier();
// CHECK-NEXT:  TB1(b);
// CHECK-NEXT:  sycl::all_of_group(sycl::ext::oneapi::experimental::this_sub_group(), a);
// CHECK-NEXT:  dpct::select_from_sub_group(sycl::ext::oneapi::experimental::this_sub_group(), a, a);
// CHECK-NEXT:}
__global__ void test2() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();
  cg::sync(cta);
  TB1(b);
  __all(a);
  __shfl(a, a);
}

// CHECK: void test3() {
// CHECK-NEXT:  int a = sycl::ext::oneapi::experimental::this_nd_item<1>().get_local_id(0);
// CHECK-NEXT:  TB1(b);
// CHECK-NEXT:}
__global__ void test3() {
  int a = threadIdx.x;
  TB1(b);
}

int main() {
    test1<<<32, 32>>>();

    test2<<<dim3(32,32,32),dim3(32,32,32)>>>();

    test3<<<32,32>>>();
}
