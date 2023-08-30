void test(void **ppv, size_t s) {
  // Start
  cudaMallocHost(ppv /*void ***/, s /*size_t*/);
  // End
}
