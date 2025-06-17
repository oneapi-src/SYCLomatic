// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int *dst, int (&thread_data)[4], int end) {
  // Start
  __shared__ typename cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>::TempStorage temp_storage;
  cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>(temp_storage).Store(dst/*int **/, thread_data/*int(&)[4]*/);
  cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>(temp_storage).Store(dst/*int **/, thread_data/*int(&)[4]*/, end/*int*/);
  // End
}
