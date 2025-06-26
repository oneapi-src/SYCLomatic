// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int *src, int (&thread_data)[4], int end, int default_value) {
  // Start
  __shared__ typename cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>::TempStorage temp_storage;
  cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/);
  cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/, end/*int*/);
  cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/, end/*int*/, default_value/*int*/);
  // End
}
