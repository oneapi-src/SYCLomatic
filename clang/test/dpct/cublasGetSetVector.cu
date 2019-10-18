// RUN: dpct -extra-arg-before=-std=c++14 -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasGetSetVector.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
// CHECK: using queue_p = cl::sycl::queue *;

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
  // CHECK: queue_p stream;
  cudaStream_t stream;

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::dpct_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0]))*(1),dpct::device_to_host), 0);
  status = cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

#define INCX_MARCO 1
  const int ConstIncy = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::dpct_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0]))*(INCX_MARCO),dpct::host_to_device), 0);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, INCX_MARCO, d_A, ConstIncy);

  // CHECK: dpct::dpct_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0]))*(1),dpct::device_to_host);
  cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

  // CHECK: dpct::dpct_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0]))*(1),dpct::host_to_device);
  cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1016:{{[0-9]+}}: The cublasGetVector was not migrated, because parameter 2 does not equal to parameter 1. Rewrite this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);
  cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);

#define INCY_MARCO 2
  // CHECK: /*
  // CHECK-NEXT: DPCT1016:{{[0-9]+}}: The cublasSetVector was not migrated, because parameter 1 does not equal to parameter INCY_MARCO. Rewrite this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, INCY_MARCO);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, INCY_MARCO);

  const int ConstIncx = 2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter ConstIncx equals to parameter INCY_MARCO but greater than 1, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::dpct_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0]))*(ConstIncx),dpct::host_to_device), 0);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncx, d_A, INCY_MARCO);

  int incx = 1;
  int incy = 1;

  // CHECK: /*
  // CHECK-NEXT: DPCT1016:{{[0-9]+}}: The cublasSetVector was not migrated, because parameter(s) incx and/or incy could not be evaluated. Rewrite this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, incx, d_A, incy);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, incx, d_A, incy);

  const int ConstIncxNE = incx;
  const int ConstIncyNE = incy;
  // CHECK: /*
  // CHECK-NEXT: DPCT1016:{{[0-9]+}}: The cublasSetVector was not migrated, because parameter(s) ConstIncxNE and/or ConstIncyNE could not be evaluated. Rewrite this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncxNE, d_A, ConstIncyNE);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncxNE, d_A, ConstIncyNE);

  const int ConstIncxT = 1;
  const int ConstIncyT = 1;
  constexpr int ConstExprIncx = 3;
  constexpr int ConstExprIncy = 3;
  // CHECK: /*
  // CHECK-NEXT: DPCT1016:{{[0-9]+}}: The cublasSetVector was not migrated, because parameter(s) foo(incx) and/or foo(incy) could not be evaluated. Rewrite this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSetVector(N, sizeof(h_A[0]), h_A, foo(incx), d_A, foo(incy));
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(incx), d_A, foo(incy));

  // CHECK: dpct::dpct_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0]))*(foo(ConstIncxT)),dpct::host_to_device);
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(ConstIncxT), d_A, foo(ConstIncyT));

  // CHECK: /*
  // CHECK-NEXT: DPCT1018:{{[0-9]+}}: The cublasGetVector was migrated, but due to parameter foo(ConstExprIncx) equals to parameter ConstExprIncy but greater than 1, the generated code performance may be sub-optimal.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::dpct_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0]))*(foo(ConstExprIncx)),dpct::device_to_host);
  cublasGetVector(N, sizeof(h_A[0]), h_A, foo(ConstExprIncx), d_A, ConstExprIncy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::dpct_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0]))*(1),dpct::device_to_host), 0);
  // CHECK-NEXT: dpct::dpct_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0]))*(1),dpct::device_to_host);
  status = cublasGetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);
  cublasGetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::dpct_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0]))*(1),dpct::host_to_device), 0);
  // CHECK-NEXT: dpct::dpct_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0]))*(1),dpct::host_to_device);
  status = cublasSetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);
  cublasSetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  return 0;
}
