void test(cudaStreamCallback_t sc, void *pv, unsigned int u) {
  // Start
  cudaStream_t s;
  cudaStreamAddCallback(s /*cudaStream_t*/, sc /*cudaStreamCallback_t*/,
                        pv /*void **/, u /*unsigned int*/);
  // End
}
