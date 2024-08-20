// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
//RUN: dpct --out-root %T/curandEnum --format-range=none --cuda-include-path="%cuda-path/include" %s -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curandEnum/curandEnum.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/curandEnum/curandEnum.dp.cpp -o %T/curandEnum/curandEnum.dp.o %}

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

// CHECK:dpct::rng::random_mode goo(
// CHECK-NEXT:dpct::rng::random_mode b1,
// CHECK-NEXT:dpct::rng::random_mode b2,
// CHECK-NEXT:// curandOrdering_t b3,
// CHECK-NEXT:dpct::rng::random_mode b4,
// CHECK-NEXT:dpct::rng::random_mode b5
// CHECK-NEXT:// , curandOrdering_t b6
// CHECK-NEXT:) { return b1; }
curandOrdering_t goo(
    curandOrdering_t b1,
    curandOrdering_t b2,
    // curandOrdering_t b3,
    curandOrdering_t b4,
    curandOrdering_t b5
    // , curandOrdering_t b6
) { return b1; }

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

  // CHECK:dpct::rng::random_mode b1 = dpct::rng::random_mode::best;
  // CHECK-NEXT:dpct::rng::random_mode b2 = dpct::rng::random_mode::best;
  // CHECK-NEXT:// curandOrdering_t b3 = CURAND_ORDERING_PSEUDO_SEEDED;
  // CHECK-NEXT:dpct::rng::random_mode b4 = dpct::rng::random_mode::legacy;
  // CHECK-NEXT:dpct::rng::random_mode b5 = dpct::rng::random_mode::optimal;
  // CHECK-NEXT:// curandOrdering_t b6 = CURAND_ORDERING_QUASI_DEFAULT;
  curandOrdering_t b1 = CURAND_ORDERING_PSEUDO_BEST;
  curandOrdering_t b2 = CURAND_ORDERING_PSEUDO_DEFAULT;
  // curandOrdering_t b3 = CURAND_ORDERING_PSEUDO_SEEDED;
  curandOrdering_t b4 = CURAND_ORDERING_PSEUDO_LEGACY;
  curandOrdering_t b5 = CURAND_ORDERING_PSEUDO_DYNAMIC;
  // curandOrdering_t b6 = CURAND_ORDERING_QUASI_DEFAULT;

  // CHECK:goo(
  // CHECK-NEXT:  dpct::rng::random_mode::best,
  // CHECK-NEXT:  dpct::rng::random_mode::best,
  // CHECK-NEXT:  // CURAND_ORDERING_PSEUDO_SEEDED,
  // CHECK-NEXT:  dpct::rng::random_mode::legacy,
  // CHECK-NEXT:  dpct::rng::random_mode::optimal
  // CHECK-NEXT:  // , CURAND_ORDERING_QUASI_DEFAULT
  // CHECK-NEXT:);
  goo(
      CURAND_ORDERING_PSEUDO_BEST,
      CURAND_ORDERING_PSEUDO_DEFAULT,
      // CURAND_ORDERING_PSEUDO_SEEDED,
      CURAND_ORDERING_PSEUDO_LEGACY,
      CURAND_ORDERING_PSEUDO_DYNAMIC
      // , CURAND_ORDERING_QUASI_DEFAULT
  );
}
