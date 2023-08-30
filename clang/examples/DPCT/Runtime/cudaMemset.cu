void test(void *pv, int i, size_t s) {
  // Start
  cudaMemset(pv /*void **/, i /*int*/, s /*size_t*/);
  // End
}
