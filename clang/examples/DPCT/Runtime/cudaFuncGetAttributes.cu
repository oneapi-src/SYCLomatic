void test(const void *f) {
  // Start
  cudaFuncAttributes *attr;
  cudaFuncGetAttributes(attr, f /*const void **/);
  // End
}
