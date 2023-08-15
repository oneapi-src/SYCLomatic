void test(cudaStream_t s, unsigned int *f) {
  // Start
  cudaStreamGetFlags(s /*cudaStream_t*/, f /*unsigned int **/);
  // End
}
