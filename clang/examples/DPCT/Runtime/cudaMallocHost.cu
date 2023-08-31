void test(void **pHost, size_t s) {
  // Start
  cudaMallocHost(pHost /*void ***/, s /*size_t*/);
  // End
}
