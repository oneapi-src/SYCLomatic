void test(void *dst, const void *src, size_t s) {
  // Start
  cudaMemcpyKind m;
  cudaMemcpy(dst /*void **/, src /*const void **/, s /*size_t*/, m);
  // End
}
