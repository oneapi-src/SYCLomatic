// Option: --use-experimental-features=graph

void test(cudaGraphNode_t node, cudaGraphNodeType *node_type) {
  // Start
  cudaGraphNodeGetType(node /*cudaGraphNode_t*/,
                       node_type /*cudaGraphNodeType **/);
  // End
}
