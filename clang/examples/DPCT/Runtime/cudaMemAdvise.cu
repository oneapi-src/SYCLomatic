void test(const void *pv, size_t s, cudaMemoryAdvise m, int i) {
  // Start
  cudaMemAdvise(pv /*const void **/, s /*size_t*/, m /*cudaMemoryAdvise*/,
                i /*int*/);
  // End
}
