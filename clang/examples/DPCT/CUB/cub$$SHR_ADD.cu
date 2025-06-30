// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int res, int a, int b, int c) {
  // Start
  res = cub::SHR_ADD(a/*int*/, b/*int*/, c/*int*/);
  // End
}
