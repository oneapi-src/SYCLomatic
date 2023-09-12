// Option: --use-experimental-features=occupancy-calculation
void test(int *pi, const void *pFunc, int i, size_t s) {
  // Start
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      pi /*int **/, pFunc /*const void **/, i /*int*/, s /*size_t*/);
  // End
}
