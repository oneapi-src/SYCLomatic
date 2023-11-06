void test(void *pHost, size_t s, unsigned int u) {
  // Start
  cudaHostRegister(pHost /*void **/, s /*size_t*/, u /*unsigned int*/);
  // End
}
