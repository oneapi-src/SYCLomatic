void test(const void *pFunc, cudaSharedMemConfig s) {
  // Start
  cudaFuncSetSharedMemConfig(pFunc /*const void **/, s /*cudaSharedMemConfig*/);
  // End
}
