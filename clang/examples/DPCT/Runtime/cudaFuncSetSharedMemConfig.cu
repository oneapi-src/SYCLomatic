// Migration desc: The API is Removed.
void test(const void *pv, cudaSharedMemConfig s) {
  // Start
  cudaFuncSetSharedMemConfig(pv /*const void **/, s /*cudaSharedMemConfig*/);
  // End
}
