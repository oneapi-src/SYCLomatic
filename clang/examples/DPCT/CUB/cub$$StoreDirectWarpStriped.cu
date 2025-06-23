// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int id, int *data, int (&thread_data)[4]) {
  // Start
  cub::StoreDirectWarpStriped(id, data, thread_data);
  // End
}
