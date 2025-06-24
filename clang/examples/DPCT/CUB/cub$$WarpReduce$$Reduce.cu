// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int thread_data, int valid_items) {
  // Start
  __shared__ typename cub::WarpReduce<int>::TempStorage temp_storage;
  int result1 = cub::WarpReduce<int>(temp_storage).Reduce(thread_data/*int*/, cub::Min()/*ReductionOp*/);
  int result2 = cub::WarpReduce<int>(temp_storage).Reduce(thread_data/*int*/, cub::Min()/*ReductionOp*/, valid_items/*int*/);
  // End
}
