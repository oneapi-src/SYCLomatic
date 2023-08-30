void test(cudaArray_t *pa, const cudaChannelFormatDesc *pc, size_t s1,
          size_t s2, unsigned int u) {
  // Start
  cudaMallocArray(pa /*cudaArray_t **/, pc /*cudaChannelFormatDesc **/,
                  s1 /*size_t*/, s2 /*size_t*/, u /*unsigned int*/);
  // End
}
