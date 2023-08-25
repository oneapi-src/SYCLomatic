void test(const void *pv, size_t s, int i) {
  // Start
  cudaStream_t cs;
  cudaMemPrefetchAsync(pv /*const void **/, s /*size_t*/, i /*int*/,
                       cs /*cudaStream_t*/);
  // End
}
