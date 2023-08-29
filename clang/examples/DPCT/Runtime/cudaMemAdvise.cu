void test(const void *pDev, size_t s, cudaMemoryAdvise m, int i) {
  // Start
  cudaMemAdvise(pDev /*const void **/, s /*size_t*/, m /*cudaMemoryAdvise*/,
                i /*int*/);
  // End
}
