// Migration desc: The API is Removed.
void test(cudaFuncCache f) {
  // Start
  cudaDeviceSetCacheConfig(f /*cudaFuncCache*/);
  // End
}
