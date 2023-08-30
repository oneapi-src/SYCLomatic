void test(void *pv, size_t s1, const void *cpv, size_t s2, size_t s3, size_t s4,
          cudaMemcpyKind m, cudaStream_t s) {
  // Start
  cudaMemcpy2DAsync(pv /*void **/, s1 /*size_t*/, cpv /*const void **/,
                    s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
                    m /*cudaMemoryAdvise*/, s /*cudaStream_t*/);
  // End
}
