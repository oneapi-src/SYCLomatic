// Option: --use-experimental-features=occupancy-calculation

void test(int *pi, CUfunction f, int i, size_t s) {
  // Start
  cuOccupancyMaxActiveBlocksPerMultiprocessor(pi /*int **/, f /*CUfunction*/,
                                              i /*int*/, s /*size_t*/);
  // End
}
