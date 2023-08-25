void test(cudaArray_t a, size_t s1, size_t s2, cudaArray_const_t ac, size_t s3,
          size_t s4, size_t s5, cudaMemcpyKind m) {
  // Start
  cudaMemcpyArrayToArray(a /*cudaArray_t*/, s1 /*size_t*/, s2 /*size_t*/,
                         ac /*cudaArray_const_t*/, s3 /*size_t*/, s4 /*size_t*/,
                         s5 /*size_t*/, m /*cudaMemcpyKind*/);
  // End
}
