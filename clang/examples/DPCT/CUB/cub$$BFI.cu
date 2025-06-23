// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(unsigned int a, unsigned int b, unsigned int c, unsigned int bit_start, unsigned int num_bits) {
  // Start
  cub::BFI(a/*unsigned int*/, b/*unsigned int*/, c/*unsigned int*/, bit_start/*unsigned int*/, num_bits/*unsigned int*/);
  // End
}
