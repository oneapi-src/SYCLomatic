void test(void *pv, int i, size_t s) {
  // Start
  cudaStream_t cs;
  cudaMemsetAsync(pv /*void **/, i /*int*/, s /*size_t*/, cs /*cudaStream_t*/);
  // End
}
