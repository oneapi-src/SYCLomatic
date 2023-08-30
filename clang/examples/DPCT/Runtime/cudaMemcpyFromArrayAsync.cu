void test(void *pv, cudaArray_const_t a, size_t s1, size_t s2, size_t s3,
          cudaMemcpyKind m, cudaStream_t s) {
  // Start
  cudaMemcpyFromArrayAsync(pv /*void **/, a /*cudaArray_const_t*/,
                           s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/,
                           m /*cudaMemcpyKind*/, s /*cudaStream_t*/);
  // End
}
