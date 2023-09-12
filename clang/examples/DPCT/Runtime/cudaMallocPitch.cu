void test(void **pDev, size_t *pz, size_t s1, size_t s2) {
  // Start
  cudaMallocPitch(pDev /*void ***/, pz /*size_t **/, s1 /*size_t*/,
                  s2 /*size_t*/);
  // End
}
