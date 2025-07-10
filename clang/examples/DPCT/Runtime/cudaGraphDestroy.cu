// Option: --use-experimental-features=graph
#include <cuda.h>
#include <cuda_runtime.h>

void test() {
  cudaGraph_t graph;
  // clang-format off
  // Start
  cudaGraphDestroy(graph /*cudaGraph_t*/);
  // End
  // clang-format on
}

