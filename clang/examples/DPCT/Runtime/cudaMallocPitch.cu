void test(void **ppv, size_t *pz, size_t s1, size_t s2) {
  // Start
  cudaMallocPitch(ppv /*void ***/, pz /*size_t **/, s1 /*size_t*/,
                  s2 /*size_t*/);
  // End
}
