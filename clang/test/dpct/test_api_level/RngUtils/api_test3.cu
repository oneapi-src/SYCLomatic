// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/RngUtils/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/RngUtils/api_test3_out/MainSourceFiles.yaml | wc -l > %T/RngUtils/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/RngUtils/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/RngUtils/api_test3_out

// CHECK: 32
// TEST_FEATURE: RngUtils_random_engine_type
// TEST_FEATURE: RngUtils_create_host_rng
// TEST_FEATURE: RngUtils_typedef_host_rng_ptr

#include "curand.h"

int main() {
  curandGenerator_t rng;
  curandCreateGeneratorHost(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  return 0;
}
