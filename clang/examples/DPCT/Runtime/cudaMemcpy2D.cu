void test(void *pv, size_t s1, const void *cpv, size_t s2, size_t s3, size_t s4,
          cudaMemcpyKind m) {
  // Start
  cudaMemcpy2D(pv /*void **/, s1 /*size_t*/, cpv /*const void **/,
               s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
               m /*cudaMemcpyKind*/);
  // End
}
