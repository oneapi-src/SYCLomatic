// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none --no-dpcpp-extensions=bfloat16 -out-root %T/math/bfloat16/bfloat16_not_support %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/bfloat16/bfloat16_not_support/bfloat16_not_support.dp.cpp

#include "cuda_bf16.h"

__global__ void f() {
  // CHECK: __nv_bfloat16 bf16;
  __nv_bfloat16 bf16;
  // CHECK: __nv_bfloat162 bf162;
  __nv_bfloat162 bf162;
  float f;
  float2 f2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hadd_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hadd_sat(bf16,bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hfma_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hfma_sat(bf16,bf16,bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hmul_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hmul_sat(bf16,bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hsub_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hsub_sat(bf16,bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __habs2 is not supported.
  // CHECK-NEXT: */
  bf162 = __habs2(bf162);
  // CHECK: f2 = sycl::float2(bf162[0], bf162[1]);
  f2 = __bfloat1622float2(bf162);
  // CHECK: f = static_cast<float>(bf16);
  f = __bfloat162float(bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float22bfloat162_rn is not supported.
  // CHECK-NEXT: */
  bf162 = __float22bfloat162_rn(f2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float2bfloat16 is not supported.
  // CHECK-NEXT: */
  bf16 = __float2bfloat16(f);
}

int main() { return 0; }
