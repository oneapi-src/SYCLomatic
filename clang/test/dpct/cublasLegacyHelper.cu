// RUN: dpct --no-cl-namespace-inline --format-range=none --usm-level=none -out-root %T/cublasLegacyHelper %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasLegacyHelper/cublasLegacyHelper.dp.cpp --match-full-lines %s
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <oneapi/mkl.hpp>
// CHECK: #include <complex>
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

// CHECK: /*
// CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasInit was replaced with 0 because this call is redundant in SYCL.
// CHECK-NEXT: */
// CHECK-NEXT: #define MACRO_A 0
#define MACRO_A cublasInit()

#define MACRO_B(status) (status)

// CHECK: /*
// CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT: */
// CHECK-NEXT: #define MACRO_C(pointer) status = (dpct::dpct_free(pointer), 0)
#define MACRO_C(pointer) status = cublasFree(pointer)

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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: foo(0, 1, 3, 7, 8, 11, 13, 14, 15, 16);
  foo(CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR);
  // CHECK: bar(0, 1, 3, 7, 8, 11, 13, 14, 15, 16);
  bar(CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR);

  // CHECK: int status;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasInit was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = 0;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasInit was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (status != 0) {
  // CHECK-NEXT:   fprintf(stderr, "!!!! CUBLAS initialization error\n");
  // CHECK-NEXT:   return EXIT_FAILURE;
  // CHECK-NEXT: }
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
  // CHECK-NEXT: a = sizeof(dpct::queue_ptr);
  // CHECK-NEXT: a = sizeof(cl::sycl::float2);
  // CHECK-NEXT: a = sizeof(cl::sycl::double2);
  int a = sizeof(cublasStatus);
  a = sizeof(cublasStatus_t);
  a = sizeof(cublasHandle_t);
  a = sizeof(cuComplex);
  a = sizeof(cuDoubleComplex);

  // CHECK: dpct::queue_ptr stream1;
  // CHECK-NEXT: stream1 = dev_ct1.create_queue();
  // CHECK-NEXT: dev_ct1.set_saved_queue(stream1);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cublasErrCheck((dev_ct1.set_saved_queue(stream1), 0));
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cublasSetKernelStream(stream1);
  cublasErrCheck(cublasSetKernelStream(stream1));

  float *d_A = NULL;
  int n = 10;
  int elemSize = 4;

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  // CHECK-NEXT: status = (d_A = (float *)dpct::dpct_malloc((n)*(elemSize)), 0);
  // CHECK-NEXT: d_A = (float *)dpct::dpct_malloc((n)*(elemSize));
  status = cublasAlloc(n, elemSize, (void **)&d_A);
  cublasAlloc(n, elemSize, (void **)&d_A);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2((d_A = (float *)dpct::dpct_malloc((n)*(elemSize)), 0));
  foo2(cublasAlloc(n, elemSize, (void **)&d_A));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (dpct::dpct_free(d_A), 0);
  // CHECK-NEXT: dpct::dpct_free(d_A);
  status = cublasFree(d_A);
  cublasFree(d_A);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2((dpct::dpct_free(d_A), 0));
  foo2(cublasFree(d_A));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MACRO_B((dpct::dpct_free(d_A), 0));
  MACRO_B(cublasFree(d_A));

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasGetError was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: MACRO_B(0);
  MACRO_B(cublasGetError());

  MACRO_C(d_A);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasGetError was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasGetError was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = 0;
  cublasGetError();
  status = cublasGetError();

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasGetError was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2(0);
  foo2(cublasGetError());

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasShutdown was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2(0);
  foo2(cublasShutdown());

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasInit was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: foo2(0);
  foo2(cublasInit());

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cublasShutdown was replaced with 0 because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = 0;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasShutdown was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: return 0;
  status = cublasShutdown();
  cublasShutdown();
  return 0;
}

