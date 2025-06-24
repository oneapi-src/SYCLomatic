// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int res, int *data) {
  // Start
  res = cub::ThreadLoad<cub::LOAD_CA>(data/*int **/);
  // End
}
