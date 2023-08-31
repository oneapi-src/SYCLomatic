// Migration desc: The API is Removed.
void test(const void *pFunc, cudaSharedMemConfig s) {
  // Start
  cudaFuncSetSharedMemConfig(pFunc /*const void **/, s /*cudaSharedMemConfig*/);
  // End
}
