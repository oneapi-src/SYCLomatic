void test(void*dst, int dstDevice, const void*src, int srcDevice, size_t count, cudaStream_t stream) {
  // Start
  cudaMemcpyPeerAsync(dst/*void**/, dstDevice/*int*/, src/*const void**/, srcDevice/*int*/, count/*size_t*/, stream/*cudaStream_t*/);
  // End
}
