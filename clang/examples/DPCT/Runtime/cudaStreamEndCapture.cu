void test(cudaStream_t s, cudaGraph_t *pg) {
  // Start
  cudaStreamEndCapture(s /*cudaStream_t*/, pg /*cudaGraph_t **/);
  // End
}
