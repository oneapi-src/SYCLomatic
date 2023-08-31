void test(size_t s1, size_t s2, const void *src, size_t s3) {
  // Start
  cudaArray_t dst;
  cudaMemcpyKind m;
  cudaMemcpyToArray(dst, s1 /*size_t*/, s2 /*size_t*/, src /*const void **/,
                    s3 /*size_t*/, m);
  // End
}
