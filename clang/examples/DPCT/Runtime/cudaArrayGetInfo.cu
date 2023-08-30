void test(cudaChannelFormatDesc *c, cudaExtent *e, unsigned int *u) {
  // Start
  cudaArray_t a;
  cudaArrayGetInfo(c /*cudaChannelFormatDesc **/, e /*cudaExtent **/,
                   u /*unsigned int **/, a /*cudaArray_t*/);
  // End
}
