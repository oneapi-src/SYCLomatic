void test(double *pd, size_t s) {
  // Start
  curandGenerator_t g;
  curandGenerateUniformDouble(g /*curandGenerator_t*/, pd /*double **/,
                              s /*size_t*/);
  // End
}
