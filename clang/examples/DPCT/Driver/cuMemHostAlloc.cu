void test(void **ppv, size_t s, unsigned int u) {
  // Start
  cuMemHostAlloc(ppv /*void ***/, s /*size_t*/, u /*unsigned int*/);
  // End
}
