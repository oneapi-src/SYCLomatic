// RUN: dpct --format-range=none --usm-level=none -extra-arg-before=-std=c++14 -out-root %T/cublasGetSetVector %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasGetSetVector/cublasGetSetVector.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr int foo(int i) {
  return i;
}

int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  status = cublasCreate(&handle);
  int N = 275;
  float *h_C;
  float *d_C;
  float *h_A;
  float *d_A;
  // CHECK: dpct::queue_ptr stream;
  cudaStream_t stream;

  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 1, 1, N, sizeof(h_C[0])));
  status = cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

#define INCX_MARCO 1
  const int ConstIncy = 1;
  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_A, (void*)h_A, ConstIncy, INCX_MARCO, 1, N, sizeof(h_A[0])));
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, INCX_MARCO, d_A, ConstIncy);

  // CHECK: dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 1, 1, N, sizeof(h_C[0]));
  cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

  // CHECK: dpct::matrix_mem_copy((void*)d_A, (void*)h_A, 1, 1, 1, N, sizeof(h_A[0]));
  cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasGetVector was migrated, but due to parameter 2 does not equal to parameter 1, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 2, 1, N, sizeof(h_C[0]));
  cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);

#define INCY_MARCO 2
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter 1 does not equal to parameter INCY_MARCO, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_A, (void*)h_A, INCY_MARCO, 1, 1, N, sizeof(h_A[0])));
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, INCY_MARCO);

  const int ConstIncx = 2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter ConstIncx equals to parameter INCY_MARCO but greater than 1, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_A, (void*)h_A, INCY_MARCO, ConstIncx, 1, N, sizeof(h_A[0])));
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncx, d_A, INCY_MARCO);

  int incx = 1;
  int incy = 1;

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter(s) incx and/or incy could not be evaluated, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_A, (void*)h_A, incy, incx, 1, N, sizeof(h_A[0])));
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, incx, d_A, incy);

  const int ConstIncxNE = incx;
  const int ConstIncyNE = incy;
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter(s) ConstIncxNE and/or ConstIncyNE could not be evaluated, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)d_A, (void*)h_A, ConstIncyNE, ConstIncxNE, 1, N, sizeof(h_A[0])));
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncxNE, d_A, ConstIncyNE);

  const int ConstIncxT = 1;
  const int ConstIncyT = 1;
  constexpr int ConstExprIncx = 3;
  constexpr int ConstExprIncy = 3;
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter(s) foo(incx) and/or foo(incy) could not be evaluated, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::matrix_mem_copy((void*)d_A, (void*)h_A, foo(incy), foo(incx), 1, N, sizeof(h_A[0]));
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(incx), d_A, foo(incy));

  // CHECK: dpct::matrix_mem_copy((void*)d_A, (void*)h_A, foo(ConstIncyT), foo(ConstIncxT), 1, N, sizeof(h_A[0]));
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(ConstIncxT), d_A, foo(ConstIncyT));

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasGetVector was migrated, but due to parameter foo(ConstExprIncx) equals to parameter ConstExprIncy but greater than 1, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::matrix_mem_copy((void*)d_A, (void*)h_A, ConstExprIncy, foo(ConstExprIncx), 1, N, sizeof(h_A[0]));
  cublasGetVector(N, sizeof(h_A[0]), h_A, foo(ConstExprIncx), d_A, ConstExprIncy);

  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 1, 1, N, sizeof(h_C[0]), dpct::automatic, *stream, true));
  // CHECK-NEXT: dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 1, 1, N, sizeof(h_C[0]), dpct::automatic, *stream, true);
  status = cublasGetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);
  cublasGetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  // CHECK: status = DPCT_CHECK_ERROR(dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 1, 1, N, sizeof(h_C[0]), dpct::automatic, *stream, true));
  // CHECK-NEXT: dpct::matrix_mem_copy((void*)h_C, (void*)d_C, 1, 1, 1, N, sizeof(h_C[0]), dpct::automatic, *stream, true);
  status = cublasSetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);
  cublasSetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  return 0;
}

