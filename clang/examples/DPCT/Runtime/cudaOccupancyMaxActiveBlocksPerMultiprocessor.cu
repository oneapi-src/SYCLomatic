// Option: --use-experimental-features=occupancy-calculation
void test(int *pi, const void *pv, int i, size_t s) {
  // Start
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      pi /*int **/, pv /*const void **/, i /*int*/, s /*size_t*/);
  // End
}
