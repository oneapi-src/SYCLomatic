void test(void *pDev, size_t s1, int i, size_t s2, size_t s3) {
  // Start
  cudaStream_t s;
  cudaMemset2DAsync(pDev /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
                    s3 /*size_t*/, s);
  // End
}
