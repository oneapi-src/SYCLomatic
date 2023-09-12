void test(cudaArray_t *pa, size_t s1, size_t s2, unsigned int u) {
  // Start
  const cudaChannelFormatDesc *pc;
  cudaMallocArray(pa /*cudaArray_t **/, pc, s1 /*size_t*/, s2 /*size_t*/, u);
  // End
}
