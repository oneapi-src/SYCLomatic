// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none -out-root %T/math/cuda-math-syclcompat %s -use-syclcompat --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-syclcompat/cuda-math-syclcompat.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST %T/math/cuda-math-syclcompat/cuda-math-syclcompat.dp.cpp -o %T/math/cuda-math-syclcompat/cuda-math-syclcompat.dp.o %}

#include "cuda_bf16.h"
#include "cuda_fp16.h"

#ifndef BUILD_TEST
__global__ void kernelFuncBfloat162Arithmetic() {
  __nv_bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  __nv_bfloat162 bf162, bf162_1, bf162_2, bf162_3;
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hadd2_sat" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hadd2_sat(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hcmadd" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hcmadd(bf162_1, bf162_2, bf162_3);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hfma2_sat" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hfma2_sat(bf162_1, bf162_2, bf162_3);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hmul2_sat" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hmul2_sat(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hsub2_sat" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hsub2_sat(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__heq2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __heq2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hequ2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hequ2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hge2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hge2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hgeu2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hgeu2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hgt2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hgt2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hgtu2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hgtu2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hle2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hle2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hleu2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hleu2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hlt2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hlt2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hltu2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hltu2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hmax2_nan" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hmax2_nan(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hmin2_nan" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hmin2_nan(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hne2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hne2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hneu2_mask" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf162 = __hneu2_mask(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hfma_relu is not supported.
  // CHECK-NEXT: */
  bf16 = __hfma_relu(bf16_1, bf16_2, bf16_3);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hmax_nan" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf16 = __hmax_nan(bf16_1, bf16_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hmin_nan" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  bf16 = __hmin_nan(bf16_1, bf16_2);

  __half2 h2, h2_1, h2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "__hcmadd" is not supported with SYCLcompat currently, please adjust the code manually.
  // CHECK-NEXT: */
  h2_2 = __hcmadd(h2, h2_1, h2_2);
}
#endif