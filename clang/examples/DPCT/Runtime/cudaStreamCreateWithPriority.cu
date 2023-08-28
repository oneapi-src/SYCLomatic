void test(cudaStream_t *ps, unsigned int u, int i) {
  // Start
  cudaStreamCreateWithPriority(ps /*cudaStream_t **/, u /*unsigned int*/,
                               i /*int*/);
  // End
}
