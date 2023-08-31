void test(void *dst, size_t s1, const void *src, size_t s2, size_t s3,
          size_t s4) {
  // Start
  cudaMemcpyKind m;
  cudaStream_t s;
  cudaMemcpy2DAsync(dst /*void **/, s1 /*size_t*/, src /*const void **/,
                    s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/, m, s);
  // End
}
