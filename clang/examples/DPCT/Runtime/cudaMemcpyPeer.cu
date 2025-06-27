void test(void*dst, int dstDevice, const void*src, int srcDevice, size_t count) {
  // Start
  cudaMemcpyPeer(dst/*void**/, dstDevice/*int*/, src/*const void**/, srcDevice/*int*/, count/*size_t*/);
  // End
}
