// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int data) {
  // Start
  __shared__ typename cub::BlockReduce<int, 4>::TempStorage temp_storage;
  cub::BlockReduce<int, 4>(temp_storage).Sum(data/*int*/);
  // End
}
