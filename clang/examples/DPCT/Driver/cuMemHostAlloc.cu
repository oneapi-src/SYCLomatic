void test(void **pHost, size_t s, unsigned int u) {
  // Start
  cuMemHostAlloc(pHost /*void ***/, s /*size_t*/, u /*unsigned int*/);
  // End
}
