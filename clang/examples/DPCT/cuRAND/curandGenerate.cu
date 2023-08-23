void test(unsigned int *pu, size_t s) {
  // Start
  curandGenerator_t g;
  curandGenerate(g /*curandGenerator_t*/, pu /*unsigned int **/, s /*size_t*/);
  // End
}
