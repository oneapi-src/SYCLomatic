void test(cudaArray_t *pa, const cudaChannelFormatDesc *pc, cudaExtent e,
          unsigned int u) {
  // Start
  cudaMalloc3DArray(pa /*cudaArray_t **/, pc /*cudaChannelFormatDesc **/,
                    e /*cudaExtent*/, u /*unsigned int*/);
  // End
}
