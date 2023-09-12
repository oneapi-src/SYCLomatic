void test(void *dst, size_t s1, size_t s2, size_t s3) {
  // TODO: src's type need to be changed to cudaArray_const_t
  // Start
  cudaArray_t src;
  cudaMemcpyKind m;
  cudaMemcpyFromArray(dst /*void **/, src, s1 /*size_t*/, s2 /*size_t*/,
                      s3 /*size_t*/, m);
  // End
}
