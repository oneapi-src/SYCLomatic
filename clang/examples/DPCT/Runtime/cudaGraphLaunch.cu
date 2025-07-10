// Option: --use-experimental-features=graph

void test(cudaGraphExec_t graph_exec, cudaStream_t stream) {
  // Start
  cudaGraphLaunch(graph_exec /*cudaGraphExec_t*/, stream /*cudaStream_t*/);
  // End
}
