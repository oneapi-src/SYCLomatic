void test(cudaStreamCallback_t sc, void *pData, unsigned int u) {
  // Start
  cudaStream_t s;
  cudaStreamAddCallback(s /*cudaStream_t*/, sc /*cudaStreamCallback_t*/,
                        pData /*void **/, u /*unsigned int*/);
  // End
}
