// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int input, unsigned int bit_start, unsigned int num_bits) {
  // Start
  cub::BFE(input/*int*/, bit_start/*unsigned int*/, num_bits/*unsigned int*/);
  // End
}
