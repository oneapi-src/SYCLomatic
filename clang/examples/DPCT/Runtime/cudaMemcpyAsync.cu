void test(void *pv, const void *cpv, size_t s, cudaMemcpyKind m) {
  // Start
  cudaStream_t cs;
  cudaMemcpyAsync(pv /*void **/, cpv /*const void **/, s /*size_t*/,
                  m /*cudaMemcpyKind*/, cs /*cudaStream_t*/);
  // End
}
