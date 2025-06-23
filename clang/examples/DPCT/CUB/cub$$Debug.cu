// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(cudaError_t e) {
  // Start
  cub::Debug(e, __FILE__, __LINE__);
  // End
}
