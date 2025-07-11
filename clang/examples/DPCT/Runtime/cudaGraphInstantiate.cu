// Option: --use-experimental-features=graph

void test(cudaGraphExec_t *graph_exec, cudaGraph_t graph, cudaGraphNode_t *node,
          char *buffer, size_t buffer_size) {
  // Start
  cudaGraphInstantiate(graph_exec /*cudaGraphExec_t **/, graph /*cudaGraph_t*/,
                       node /*cudaGraphNode_t **/, buffer /*char **/,
                       buffer_size /*size_t*/);
  // End
}
