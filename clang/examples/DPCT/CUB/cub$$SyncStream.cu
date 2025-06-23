// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(cudaStream_t s) {
  // Start
  cub::SyncStream(s/*cudaStream_t*/);
  // End
}
