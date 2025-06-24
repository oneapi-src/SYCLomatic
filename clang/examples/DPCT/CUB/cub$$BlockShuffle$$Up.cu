// clang-format off
#include <cstddef>
#include <cub/cub.cuh>
#include <cub/block/block_shuffle.cuh>

__device__ void test(int (&input)[4], int (&output)[4]) {
  // Start
  __shared__ typename cub::BlockShuffle<int, 128>::TempStorage temp_storage;
  cub::BlockShuffle<int, 128>(temp_storage).Up(input/*int*/, output/*int &*/);
  // End
}
