void test(CUlimit l, size_t s) {
  // Start
  cuCtxSetLimit(l /*CUlimit*/, s /*size_t*/);
  // End
}
