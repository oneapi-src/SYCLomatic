void test(void **ppv) {
  // Start
  const void *pv;
  cudaGetSymbolAddress(ppv /*void ***/, pv /*const void **/);
  // End
}
