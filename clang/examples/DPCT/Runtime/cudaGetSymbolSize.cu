void test(size_t *s) {
  // Start
  const void *pv;
  cudaGetSymbolSize(s /*size_t **/, pv /*const void **/);
  // End
}
