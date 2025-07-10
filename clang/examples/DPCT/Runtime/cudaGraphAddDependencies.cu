// Option: --use-experimental-features=graph
#include <cuda.h>
#include <cuda_runtime.h>

void test() {
  cudaGraph_t graph;
  cudaGraphNode_t node4[10];
  cudaGraphNode_t node5[10];
  // clang-format off
  // Start
  cudaGraphAddDependencies(graph /*cudaGraph_t*/, node4 /*const cudaGraphNode_t* */, node5 /*const cudaGraphNode_t* */, 10 /*size_t*/);
  // End
  // clang-format on
}

