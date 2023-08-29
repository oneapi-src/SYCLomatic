void test(cudaStream_t s, int *pi) {
  // Start
  cudaStreamGetPriority(s /*cudaStream_t*/, pi /*int **/);
  // End
}
