// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int (&thread_data)[4]) {
  // Start
  __shared__ typename cub::BlockRadixSort<int, 128, 4>::TempStorage temp_storage;
  cub::BlockRadixSort<int, 128, 4>(temp_storage).Sort(thread_data/*int(&)[4]*/);
  // End
}
