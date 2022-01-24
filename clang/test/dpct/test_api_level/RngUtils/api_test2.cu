// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/RngUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/RngUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/RngUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/RngUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/RngUtils/api_test2_out

// CHECK: 4
// TEST_FEATURE: RngUtils_rng_generator
// TEST_FEATURE: RngUtils_rng_generator_get_engine

#include <curand_kernel.h>
__device__ void foo() {
  curandStatePhilox4_32_10_t rng;
  curand_init(1, 2, 3, &rng);
  skipahead (1, &rng);
}

int main() {
  return 0;
}
