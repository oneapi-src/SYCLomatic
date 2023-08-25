void test(void *pv, size_t s1, int i, size_t s2, size_t s3, cudaStream_t s) {
  // Start
  cudaMemset2DAsync(pv /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
                    s3 /*size_t*/, s /*cudaStream_t*/);
  // End
}
