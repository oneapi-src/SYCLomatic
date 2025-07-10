// Option: --use-experimental-features=graph

void test(cudaGraphExec_t graph_exec, cudaGraph_t graph, cudaGraphNode_t *node,
          cudaGraphExecUpdateResult *result) {
  // Start
  cudaGraphExecUpdate(graph_exec /*cudaGraphExec_t*/, graph /*cudaGraph_t*/,
                      node /*cudaGraphNode_t **/,
                      result /*cudaGraphExecUpdateResult **/);
  // End
}
