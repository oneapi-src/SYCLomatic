void test(void **pDev, size_t s, unsigned int u) {
  // Start
  cudaMallocManaged(pDev /*void ***/, s /*size_t*/, u /*unsigned int*/);
  // End
}
