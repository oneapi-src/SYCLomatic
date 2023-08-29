// Migration desc: The API is Removed.
void test(cudaLimit l, size_t s) {
  // Start
  cudaThreadSetLimit(l /*cudaLimit*/, s /*size_t*/);
  // End
}
