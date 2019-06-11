// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasLegacyHelper.sycl.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <mkl_blas_sycl.hpp>
// CHECK-NEXT: #include <mkl_lapack_sycl.hpp>
// CHECK-NEXT: #include <sycl_types.hpp>
// CHECK: #include <complex>
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

// CHECK: #define MACRO_A 0
#define MACRO_A cublasInit()

#define MACRO_B(status) (status)

// CHECK: #define MACRO_C(pointer) status = (syclct::sycl_free(d_A), 0)
#define MACRO_C(pointer) status = cublasFree(d_A)

void foo2(cublasStatus){}

// CHECK: void foo(int, int, int, int, int, int, int, int, int, int) {}
void foo(cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus) {}

// CHECK: void bar(int, int, int, int, int, int, int, int, int, int) {}
void bar(cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t) {}

// CHECK: int foo(int m, int n) {
cublasStatus foo(int m, int n) {
  return CUBLAS_STATUS_SUCCESS;
}

int main() {
  // CHECK: foo(0, 1, 3, 7, 8, 11, 13, 14, 15, 16);
  foo(CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR);
  // CHECK: bar(0, 1, 3, 7, 8, 11, 13, 14, 15, 16);
  bar(CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR);

  // CHECK: int status;
  // CHECK-NEXT: status = 0;
  // CHECK-NEXT: if (status != 0) {
  cublasStatus status;
  status = cublasInit();
  cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  status = MACRO_A;

  // CHECK: int a = sizeof(int);
  // CHECK-NEXT: a = sizeof(int);
  // CHECK-NEXT: a = sizeof(cl::sycl::queue);
  // CHECK-NEXT: a = sizeof(std::complex<float>);
  // CHECK-NEXT: a = sizeof(std::complex<double>);
  int a = sizeof(cublasStatus);
  a = sizeof(cublasStatus_t);
  a = sizeof(cublasHandle_t);
  a = sizeof(cuComplex);
  a = sizeof(cuDoubleComplex);

  float *d_A = NULL;
  int n = 10;
  int elemSize = 4;

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  //CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_malloc((void **)&d_A, (n)*(elemSize)), 0);
  // CHECK-NEXT: syclct::sycl_malloc((void **)&d_A, (n)*(elemSize));
  status = cublasAlloc(n, elemSize, (void **)&d_A);
  cublasAlloc(n, elemSize, (void **)&d_A);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2((syclct::sycl_malloc((void **)&d_A, (n)*(elemSize)), 0));
  foo2(cublasAlloc(n, elemSize, (void **)&d_A));

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (syclct::sycl_free(d_A), 0);
  // CHECK-NEXT: syclct::sycl_free(d_A);
  status = cublasFree(d_A);
  cublasFree(d_A);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2((syclct::sycl_free(d_A), 0));
  foo2(cublasFree(d_A));

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: MACRO_B((syclct::sycl_free(d_A), 0));
  MACRO_B(cublasFree(d_A));

  // CHECK: MACRO_B(0);
  MACRO_B(cublasGetError());

  MACRO_C(d_A);

  // CHECK: status = 0;
  cublasGetError();
  status = cublasGetError();

  // CHECK: foo2(0);
  foo2(cublasGetError());

  // CHECK: foo2(0);
  foo2(cublasShutdown());

  // CHECK: foo2(0);
  foo2(cublasInit());

  // CHECK: status = 0;
  // CHECK-NEXT: return 0;
  status = cublasShutdown();
  cublasShutdown();
  return 0;
}