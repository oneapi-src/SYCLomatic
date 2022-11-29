// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/RngUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/RngUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/RngUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/RngUtils/api_test1_out/count.txt --match-full-lines %s -check-prefix=FEATURE_NUMBER
// RUN: FileCheck --input-file %T/RngUtils/api_test1_out/api_test1.dp.cpp --match-full-lines %s -check-prefix=CODE
// RUN: rm -rf %T/RngUtils/api_test1_out

// FEATURE_NUMBER: 5
// TEST_FEATURE: RngUtils_rng_generator
// TEST_FEATURE: RngUtils_rng_generator_generate
// TEST_FEATURE: RngUtils_non_local_include_dependency

// CODE: // AAA
// CODE-NEXT:#define DPCT_USM_LEVEL_NONE
// CODE-NEXT:#include <sycl/sycl.hpp>
// CODE-NEXT:#include <dpct/dpct.hpp>
// CODE-NEXT:#include <oneapi/mkl/rng/device.hpp>
// CODE-NEXT:// BBB

// AAA
#include <curand_kernel.h>
// BBB

__device__ void foo() {
  curandStatePhilox4_32_10_t rng;
  curand_init(1, 2, 3, &rng);
  curand(&rng);
}

int main() {
  return 0;
}
