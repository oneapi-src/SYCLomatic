void test(cudaArray_t *pa, cudaExtent e, unsigned int u) {
  // Start
  const cudaChannelFormatDesc *pc;
  cudaMalloc3DArray(pa /*cudaArray_t **/, pc, e /*cudaExtent*/, u);
  // End
}
