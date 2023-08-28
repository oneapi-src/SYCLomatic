// Migration desc: The API is Removed.
void test(const void *pv, cudaFuncCache f) {
  // Start
  cudaFuncSetCacheConfig(pv /*const void **/, f /*cudaFuncCache*/);
  // End
}
