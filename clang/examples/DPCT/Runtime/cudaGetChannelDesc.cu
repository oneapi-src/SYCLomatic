void test(cudaChannelFormatDesc *pc) {
  // Start
  cudaArray_const_t a;
  cudaGetChannelDesc(pc /*cudaChannelFormatDesc **/, a /*cudaArray_const_t*/);
  // End
}
