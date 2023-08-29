// Migration desc: The API is Removed.
void test(enum cudaFuncCache *pf) {
  // Start
  cudaDeviceGetCacheConfig(pf /*enum cudaFuncCache **/);
  // End
}
