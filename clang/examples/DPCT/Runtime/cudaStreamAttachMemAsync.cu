void test(cudaStream_t s, void *pDev, size_t st, unsigned int u) {
  // Start
  cudaStreamAttachMemAsync(s /*cudaStream_t*/, pDev /*void **/, st /*size_t*/,
                           u /*unsigned int*/);
  // End
}
