// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(cudaError_t e, const char* filename, int line) {
  // Start
  cub::Debug(e/*cudaError_t*/, filename/*const char**/, line/*int*/);
  // End
}
