// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int thread_data) {
  // Start
  __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
  cub::WarpScan<int>(temp_storage).Broadcast(thread_data/*int*/, 0/*unsigned int*/);
  // End
}
