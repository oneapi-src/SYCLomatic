// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int thread_data) {
  // Start
  __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
  cub::WarpScan<int>(temp_storage).InclusiveScan(thread_data/*int*/, thread_data/*int &*/, cub::Sum()/*ScanOp*/);
  // End
}
