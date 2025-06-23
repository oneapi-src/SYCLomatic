// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int result, unsigned int a, unsigned int b, unsigned int c) {
  // Start
  result = cub::IADD3(a/*unsigned int*/, b/*unsigned int*/, c/*unsigned int*/);
  // End
}
