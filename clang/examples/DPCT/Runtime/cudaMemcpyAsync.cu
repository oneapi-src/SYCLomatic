void test(void *dst, const void *src, size_t s, cudaMemcpyKind m) {
  // Start
  cudaStream_t cs;
  cudaMemcpyAsync(dst /*void **/, src /*const void **/, s /*size_t*/,
                  m /*cudaMemcpyKind*/, cs);
  // End
}
