// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int (&thread_data)[4], int (&thread_rank)[4]) {
  // Start
  __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
  cub::BlockExchange<int, 128, 4>(temp_storage).ScatterToStriped(thread_data/*int(&)[4]*/, thread_rank/*int(&)[4]*/);
  // End
}