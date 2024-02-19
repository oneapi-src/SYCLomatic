// RUN: dpct --format-range=none -out-root %T/ldg %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/ldg/ldg.dp.cpp
#include "cuda_bf16.h"
#include "cuda_fp16.h"
// CHECK: #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>

// CHECK: void test_ldg_tex_cache_read(int *deviceArray){
// CHECK-NEXT: float f1;
// CHECK-NEXT: double d;
// CHECK-NEXT: sycl::float2 *f2;
// CHECK-NEXT: sycl::half h1;
// CHECK-NEXT: sycl::half2 *h2;
// CHECK-NEXT: sycl::uchar4 u4;
// CHECK-NEXT: sycl::ulong2 *ull2;
// CHECK-NEXT: sycl::ext::oneapi::bfloat16 nvbf1;
// CHECK-NEXT: sycl::marray<sycl::ext::oneapi::bfloat16, 2> nvbf2;
__global__ void test_ldg_tex_cache_read(int *deviceArray){
  float f1;
  double d;
  float2 *f2;
  __half h1;
  __half2 *h2;
  uchar4 u4;
  ulonglong2 *ull2;
  __nv_bfloat16 nvbf1;
  __nv_bfloat162 nvbf2;

// CHECK:  sycl::ext::oneapi::experimental::cuda::ldg(&f1);
// CHECK-NEXT:  auto cacheReadD  = sycl::ext::oneapi::experimental::cuda::ldg(&d);
// CHECK-NEXT:  sycl::ext::oneapi::experimental::cuda::ldg(f2);
// CHECK-NEXT:  auto cacheReadH1 = sycl::ext::oneapi::experimental::cuda::ldg(&h1);
// CHECK-NEXT:  sycl::ext::oneapi::experimental::cuda::ldg(h2);
// CHECK-NEXT:  sycl::ext::oneapi::experimental::cuda::ldg(&u4);
// CHECK-NEXT:  sycl::ext::oneapi::experimental::cuda::ldg(ull2);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1122:0: SYCL API call for ldg does not support __nv_bfloat16 type. You may need to adjust this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  __ldg(&nvbf1);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1122:1: SYCL API call for ldg does not support __nv_bfloat162 type. You may need to adjust this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  __ldg(&nvbf2);
  __ldg(&f1);
  auto cacheReadD  = __ldg(&d);
  __ldg(f2);
  auto cacheReadH1 = __ldg(&h1);
  __ldg(h2);
  __ldg(&u4);
  __ldg(ull2);
  __ldg(&nvbf1);
  __ldg(&nvbf2);
}
