void test(const void *pDev, size_t s, int i) {
  // Start
  cudaStream_t cs;
  cudaMemPrefetchAsync(pDev /*const void **/, s /*size_t*/, i /*int*/,
                       cs /*cudaStream_t*/);
  // End
}
