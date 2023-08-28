void test(void *pv, cudaArray_const_t a, size_t s1, size_t s2, size_t s3,
          cudaMemcpyKind m) {
  // Start
  cudaMemcpyFromArray(pv /*void **/, a /*cudaArray_const_t*/, s1 /*size_t*/,
                      s2 /*size_t*/, s3 /*size_t*/, m /*cudaMemcpyKind*/);
  // End
}
