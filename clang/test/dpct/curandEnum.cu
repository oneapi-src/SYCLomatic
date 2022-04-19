//RUN: c2s --out-root %T/curandEnum --format-range=none --cuda-include-path="%cuda-path/include" %s -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curandEnum/curandEnum.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <stdio.h>
#include <curand.h>

// CHECK:int foo(
// CHECK-NEXT:int a1,
// CHECK-NEXT:int a2,
// CHECK-NEXT:int a3,
// CHECK-NEXT:int a4,
// CHECK-NEXT:int a5,
// CHECK-NEXT:int a6,
// CHECK-NEXT:int a7,
// CHECK-NEXT:int a8,
// CHECK-NEXT:int a9,
// CHECK-NEXT:int a10,
// CHECK-NEXT:int a11,
// CHECK-NEXT:int a12,
// CHECK-NEXT:int a13) {}
curandStatus_t foo(
  curandStatus_t a1,
  curandStatus_t a2,
  curandStatus_t a3,
  curandStatus_t a4,
  curandStatus_t a5,
  curandStatus_t a6,
  curandStatus_t a7,
  curandStatus_t a8,
  curandStatus_t a9,
  curandStatus_t a10,
  curandStatus_t a11,
  curandStatus_t a12,
  curandStatus_t a13) {}

int main() {
  // CHECK:int a1 = 0;
  // CHECK-NEXT:int a2 = 100;
  // CHECK-NEXT:int a3 = 101;
  // CHECK-NEXT:int a4 = 102;
  // CHECK-NEXT:int a5 = 103;
  // CHECK-NEXT:int a6 = 104;
  // CHECK-NEXT:int a7 = 105;
  // CHECK-NEXT:int a8 = 106;
  // CHECK-NEXT:int a9 = 201;
  // CHECK-NEXT:int a10 = 202;
  // CHECK-NEXT:int a11 = 203;
  // CHECK-NEXT:int a12 = 204;
  // CHECK-NEXT:int a13 = 999;
  curandStatus_t a1 = CURAND_STATUS_SUCCESS;
  curandStatus_t a2 = CURAND_STATUS_VERSION_MISMATCH;
  curandStatus_t a3 = CURAND_STATUS_NOT_INITIALIZED;
  curandStatus_t a4 = CURAND_STATUS_ALLOCATION_FAILED;
  curandStatus_t a5 = CURAND_STATUS_TYPE_ERROR;
  curandStatus_t a6 = CURAND_STATUS_OUT_OF_RANGE;
  curandStatus_t a7 = CURAND_STATUS_LENGTH_NOT_MULTIPLE;
  curandStatus_t a8 = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
  curandStatus_t a9 = CURAND_STATUS_LAUNCH_FAILURE;
  curandStatus_t a10 = CURAND_STATUS_PREEXISTING_FAILURE;
  curandStatus_t a11 = CURAND_STATUS_INITIALIZATION_FAILED;
  curandStatus_t a12 = CURAND_STATUS_ARCH_MISMATCH;
  curandStatus_t a13 = CURAND_STATUS_INTERNAL_ERROR;


  // CHECK:foo(
  // CHECK-NEXT:  0,
  // CHECK-NEXT:  100,
  // CHECK-NEXT:  101,
  // CHECK-NEXT:  102,
  // CHECK-NEXT:  103,
  // CHECK-NEXT:  104,
  // CHECK-NEXT:  105,
  // CHECK-NEXT:  106,
  // CHECK-NEXT:  201,
  // CHECK-NEXT:  202,
  // CHECK-NEXT:  203,
  // CHECK-NEXT:  204,
  // CHECK-NEXT:  999);
  foo(
    CURAND_STATUS_SUCCESS,
    CURAND_STATUS_VERSION_MISMATCH,
    CURAND_STATUS_NOT_INITIALIZED,
    CURAND_STATUS_ALLOCATION_FAILED,
    CURAND_STATUS_TYPE_ERROR,
    CURAND_STATUS_OUT_OF_RANGE,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
    CURAND_STATUS_LAUNCH_FAILURE,
    CURAND_STATUS_PREEXISTING_FAILURE,
    CURAND_STATUS_INITIALIZATION_FAILED,
    CURAND_STATUS_ARCH_MISMATCH,
    CURAND_STATUS_INTERNAL_ERROR);
}

