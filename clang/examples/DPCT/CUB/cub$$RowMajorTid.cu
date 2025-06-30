// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int r, int dim_x, int dim_y, int dim_z) {
  // Start
  r = cub::RowMajorTid(dim_x/*int*/, dim_y/*int*/, dim_z/*int*/);
  // End
}
