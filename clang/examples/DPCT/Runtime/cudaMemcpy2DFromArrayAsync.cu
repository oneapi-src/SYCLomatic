void test(void *pv, size_t s1, cudaArray_const_t a, size_t s2, size_t s3,
          size_t s4, size_t s5, cudaMemcpyKind m, cudaStream_t s) {
  // Start
  cudaMemcpy2DFromArrayAsync(pv /*void **/, s1 /*size_t*/,
                             a /*cudaArray_const_t*/, s2 /*size_t*/,
                             s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/,
                             m /*cudaMemcpyKind*/, s /*cudaStream_t*/);
  // End
}
