void test(unsigned int u) {
  // Start
  curandGenerator_t g;
  curandSetQuasiRandomGeneratorDimensions(g /*curandGenerator_t*/,
                                          u /*unsigned int*/);
  // End
}
