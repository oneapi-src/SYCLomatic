void test(void *pDev, int i, size_t s) {
  // Start
  cudaMemset(pDev /*void **/, i /*int*/, s /*size_t*/);
  // End
}
