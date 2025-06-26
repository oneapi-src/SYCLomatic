// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int *dst, int data) {
  // Start
  cub::ThreadStore<cub::STORE_CG>(dst/*int **/, data/*int*/);
  // End
}
