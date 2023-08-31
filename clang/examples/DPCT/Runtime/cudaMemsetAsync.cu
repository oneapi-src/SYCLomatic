void test(void *pDev, int i, size_t s) {
  // Start
  cudaStream_t cs;
  cudaMemsetAsync(pDev /*void **/, i /*int*/, s /*size_t*/, cs);
  // End
}
