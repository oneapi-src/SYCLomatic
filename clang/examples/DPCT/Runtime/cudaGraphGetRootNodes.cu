// Option: --use-experimental-features=graph

void test(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *num_nodes) {
  // Start
  cudaGraphGetRootNodes(graph /*cudaGraph_t*/, nodes /*cudaGraphNode_t **/,
                        num_nodes /*size_t **/);
  // End
}
