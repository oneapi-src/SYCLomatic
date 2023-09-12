void test(void *dst, size_t s1, size_t s2, size_t s3, size_t s4, size_t s5,
          cudaMemcpyKind m) {
  // TODO: src's type need to be changed to cudaArray_const_t
  // Start
  cudaArray_t src;
  cudaStream_t s;
  cudaMemcpy2DFromArrayAsync(dst /*void **/, s1 /*size_t*/, src, s2 /*size_t*/,
                             s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/,
                             m /*cudaMemcpyKind*/, s);
  // End
}
