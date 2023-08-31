void test(void *dst, size_t s1, size_t s2, size_t s3, cudaMemcpyKind m) {
  // TODO: src's type need to be changed to cudaArray_const_t
  // Start
  cudaArray_t src;
  cudaStream_t s;
  cudaMemcpyFromArrayAsync(dst /*void **/, src, s1 /*size_t*/, s2 /*size_t*/,
                           s3 /*size_t*/, m /*cudaMemcpyKind*/, s);
  // End
}
