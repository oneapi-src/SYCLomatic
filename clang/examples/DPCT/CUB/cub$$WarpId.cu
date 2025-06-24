// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int res) {
  // Start
  res = cub::WarpId();
  // End
}
