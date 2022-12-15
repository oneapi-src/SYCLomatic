// RUN: dpct --format-range=none --report-type=all -out-root %T/cusolverHelper %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolverHelper/cusolverHelper.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cusolverDn.h>

// CHECK: #define MACRO_A cusolverDnCreate
#define MACRO_A cusolverDnCreate

// CHECK: void foo(int, int, int, int, int, int, int, int) {}
void foo(cusolverStatus_t, cusolverStatus_t, cusolverStatus_t, cusolverStatus_t, cusolverStatus_t, cusolverStatus_t, cusolverStatus_t, cusolverStatus_t) {}

// CHECK: void foo2(int){}
void foo2(cusolverStatus_t){}

// CHECK: int foo3(int m, int n)
cusolverStatus_t foo3(int m, int n)
{
    // CHECK: return 0;
    return CUSOLVER_STATUS_SUCCESS;
}

// CHECK: extern sycl::queue* cusolverH2 = NULL;
extern cusolverDnHandle_t cusolverH2 = NULL;

int main(int argc, char *argv[])
{
    // CHECK: sycl::queue* cusolverH = NULL;
    // CHECK-NEXT: int status = 0;
    // CHECK-NEXT: status = 1;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    status = CUSOLVER_STATUS_NOT_INITIALIZED;

    // CHECK: foo(0, 1, 2, 3, 4, 6, 7, 8);
    foo(CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_NOT_INITIALIZED, CUSOLVER_STATUS_ALLOC_FAILED, CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_ARCH_MISMATCH, CUSOLVER_STATUS_EXECUTION_FAILED, CUSOLVER_STATUS_INTERNAL_ERROR, CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);

    // CHECK: cusolverH = &q_ct1;
    cusolverDnCreate(&cusolverH);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = (cusolverH = &q_ct1, 0);
    status = cusolverDnCreate(&cusolverH);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = (cusolverH = &q_ct1, 0);
    status = MACRO_A(&cusolverH);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = (cusolverH = nullptr, 0);
    status = cusolverDnDestroy(cusolverH);

    // CHECK: int a = sizeof(int);
    // CHECK-NEXT: int b = sizeof(sycl::queue*);
    int a = sizeof(cublasStatus_t);
    int b = sizeof(cusolverDnHandle_t);

    cudaStream_t stream;
    // CHECK: status = (stream = cusolverH, 0);
    // CHECK: status = (cusolverH = stream, 0);
    status = cusolverDnGetStream(cusolverH, &stream);
    status = cusolverDnSetStream(cusolverH, stream);
}


