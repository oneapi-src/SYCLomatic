void test(size_t s1, size_t s2, const void *src, size_t s3, size_t s4,
          size_t s5, cudaMemcpyKind m) {
  // Start
  cudaArray_t dst;
  cudaStream_t s;
  cudaMemcpy2DToArrayAsync(dst, s1 /*size_t*/, s2 /*size_t*/,
                           src /*const void **/, s3 /*size_t*/, s4 /*size_t*/,
                           s5 /*size_t*/, m /*cudaMemcpyKind*/, s);
  // End
}
