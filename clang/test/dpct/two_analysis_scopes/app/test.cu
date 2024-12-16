// RUN: dpct --format-range=none --out-root %T/out %s --analysis-scope-path %S --analysis-scope-path %S/../deps --cuda-include-path="%cuda-path/include" --extra-arg="-I%S/../deps"
// RUN: FileCheck --match-full-lines --input-file %T/out/test.dp.cpp %s
// RUN: FileCheck --match-full-lines --input-file %T/out/test.dp.hpp %S/test.cuh
// RUN: echo "// empty" > %T/out/dep.h
// RUN: %if build_lit %{icpx -c -fsycl %T/out/test.dp.cpp -o %T/out/test.dp.o -I%T/out %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "test.dp.hpp"
// CHECK-NEXT: #include <dpct/blas_utils.hpp>
#include "test.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void foo(cublasHandle_t handle, const half *a, const half *b, half *c,
         int n, half *alpha, half *beta) {
  // CHECK: oneapi::mkl::blas::column_major::gemm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, dpct::get_value(alpha, handle->get_queue()), a, n, b, n, dpct::get_value(beta, handle->get_queue()), c, n);
  cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
              alpha, a, n, b, n, beta, c, n);
}
