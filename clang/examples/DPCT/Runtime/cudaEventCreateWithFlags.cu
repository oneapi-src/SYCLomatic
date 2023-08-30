void test(cudaEvent_t *pe, unsigned int u) {
  // Start
  cudaEventCreateWithFlags(pe /*cudaEvent_t **/, u /*unsigned int*/);
  // End
}
