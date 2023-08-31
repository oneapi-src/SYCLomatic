void test(void *pDev, size_t s1, int i, size_t s2, size_t s3) {
  // Start
  cudaMemset2D(pDev /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
             s3 /*size_t*/);
  // End
}
