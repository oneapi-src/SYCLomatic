void test(void *dst, size_t s1, const void *src, size_t s2, size_t s3,
          size_t s4) {
  // Start
  cudaMemcpyKind m;
  cudaMemcpy2D(dst /*void **/, s1 /*size_t*/, src /*const void **/,
               s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/, m);
  // End
}
