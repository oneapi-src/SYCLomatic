// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int input, int output) {
  // Start
  __shared__ typename cub::BlockScan<int, 4>::TempStorage temp_storage;
  cub::BlockScan<int, 4>(temp_storage).InclusiveScan(input/*int*/, output/*int &*/, cub::Sum()/*ScanOp*/);
  // End
}
