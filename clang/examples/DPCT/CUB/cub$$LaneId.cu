// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int result) {
  // Start
  result = cub::LaneId();
  // End
}
