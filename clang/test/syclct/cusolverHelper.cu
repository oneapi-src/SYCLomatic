// RUN: syclct -out-root %T %s -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cusolverHelper.sycl.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
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

int main(int argc, char *argv[])
{
    // CHECK: cl::sycl::queue * cusolverH = NULL;
    // CHECK-NEXT: int status = 0;
    // CHECK-NEXT: status = 1;
    cusolverDnHandle_t* cusolverH = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    status = CUSOLVER_STATUS_NOT_INITIALIZED;

    // CHECK: foo(0, 1, 2, 3, 4, 6, 7, 8);
    // CHECK-NEXT: status = 0;
    foo(CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_NOT_INITIALIZED, CUSOLVER_STATUS_ALLOC_FAILED, CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_ARCH_MISMATCH, CUSOLVER_STATUS_EXECUTION_FAILED, CUSOLVER_STATUS_INTERNAL_ERROR, CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    cusolverDnCreate(cusolverH);
    status = cusolverDnCreate(cusolverH);

    // CHECK: status = 0;
    status = MACRO_A(cusolverH);

    // CHECK: status = 0;
    status = cusolverDnDestroy(*cusolverH);

    // CHECK: int a = sizeof(int);
    // CHECK-NEXT: int b = sizeof(cl::sycl::queue);
    int a = sizeof(cublasStatus_t);
    int b = sizeof(cusolverDnHandle_t);
}
