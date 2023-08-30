void test(void *pv, const void *cpv, size_t s, cudaMemcpyKind m) {
  // Start
  cudaMemcpy(pv /*void **/, cpv /*const void **/, s /*size_t*/,
             m /*cudaMemcpyKind*/);
  // End
}
