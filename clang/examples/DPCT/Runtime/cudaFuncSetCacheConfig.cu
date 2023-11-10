void test(const void *pFunc, cudaFuncCache f) {
  // Start
  cudaFuncSetCacheConfig(pFunc /*const void **/, f /*cudaFuncCache*/);
  // End
}
