void test(cudaArray_t a, size_t s1, size_t s2, const void *pv, size_t s3,
          cudaMemcpyKind m) {
  // Start
  cudaMemcpyToArray(a /*cudaArray_t*/, s1 /*size_t*/, s2 /*size_t*/,
                    pv /*const void **/, s3 /*size_t*/, m /*cudaMemcpyKind*/);
  // End
}
