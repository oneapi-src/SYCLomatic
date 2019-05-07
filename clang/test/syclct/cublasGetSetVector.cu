// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cublasGetSetVector.sycl.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
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
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0])),syclct::device_to_host), 0);
  status = cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0])),syclct::host_to_device), 0);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 1);
  // CHECK: syclct::sycl_memcpy((void*)(h_C),(void*)(d_C),(N)*(sizeof(h_C[0])),syclct::device_to_host);
  cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);
  // CHECK: syclct::sycl_memcpy((void*)(d_A),(void*)(h_A),(N)*(sizeof(h_A[0])),syclct::host_to_device);
  cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: parameter(s) value of incx or(and) incy in cublasGetVector is not supported in Sycl. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);
  cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1016:{{[0-9]+}}: parameter(s) value of incx or(and) incy in cublasSetVector is not supported in Sycl. You may need to migrate this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 2);
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 2);
  return 0;
}
