void test(cudaChannelFormatDesc *pc) {
  // TODO: a's type need to be changed to cudaArray_const_t
  // Start
  cudaArray_t a;
  cudaGetChannelDesc(pc /*cudaChannelFormatDesc **/, a);
  // End
}
