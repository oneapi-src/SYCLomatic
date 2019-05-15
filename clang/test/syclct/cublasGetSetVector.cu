// RUN: syclct -extra-arg-before=-std=c++11 -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cublasGetSetVector.sycl.cpp --match-full-lines %s
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

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(h_C),(void*)(d_C),(N)*((sizeof(h_C[0]))+(1))-(1),syclct::device_to_host), 0);
  status = cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

#define INCX_MARCO 1
  const int ConstIncy = 1;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*((sizeof(h_A[0]))+(INCX_MARCO))-(INCX_MARCO),syclct::host_to_device), 0);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, INCX_MARCO, d_A, ConstIncy);

  // CHECK: syclct::sycl_memcpy((void*)(h_C),(void*)(d_C),(N)*((sizeof(h_C[0]))+(1))-(1),syclct::device_to_host);
  cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

  // CHECK: syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*((sizeof(h_A[0]))+(1))-(1),syclct::host_to_device);
  cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasGetVector with these parameters is not supported currently, because parameter incx does not equal to incy. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);
  cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);

#define INCY_MARCO 2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetVector with these parameters is not supported currently, because parameter incx does not equal to incy. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, INCY_MARCO);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, INCY_MARCO);

  const int ConstIncx = 2;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1018:{{[0-9]+}}: Migration of cublasSetVector with these parameters could lead performance issue by auto-migration, because parameters incx and incy do not equal to 1. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*((sizeof(h_A[0]))+(ConstIncx))-(ConstIncx),syclct::host_to_device), 0);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncx, d_A, INCY_MARCO);

  int incx = 1;
  int incy = 1;

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetVector with these parameters is not supported currently, because parameter incx or incy cannot be evaluated. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, incx, d_A, incy);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, incx, d_A, incy);

  const int ConstIncxNE = incx;
  const int ConstIncyNE = incy;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetVector with these parameters is not supported currently, because parameter incx or incy cannot be evaluated. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncxNE, d_A, ConstIncyNE);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncxNE, d_A, ConstIncyNE);

  const int ConstIncxT = 1;
  const int ConstIncyT = 1;
  constexpr int ConstExprIncx = 3;
  constexpr int ConstExprIncy = 3;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: Migration of cublasSetVector with these parameters is not supported currently, because parameter incx or incy cannot be evaluated. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasSetVector(N, sizeof(h_A[0]), h_A, foo(incx), d_A, foo(incy));
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(incx), d_A, foo(incy));

  // CHECK: syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*((sizeof(h_A[0]))+(foo(ConstIncxT)))-(foo(ConstIncxT)),syclct::host_to_device);
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(ConstIncxT), d_A, foo(ConstIncyT));

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1018:{{[0-9]+}}: Migration of cublasGetVector with these parameters could lead performance issue by auto-migration, because parameters incx and incy do not equal to 1. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*((sizeof(h_A[0]))+(foo(ConstExprIncx)))-(foo(ConstExprIncx)),syclct::device_to_host);
  cublasGetVector(N, sizeof(h_A[0]), h_A, foo(ConstExprIncx), d_A, ConstExprIncy);

  return 0;
}
