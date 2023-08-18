void test(CUdeviceptr *pd, size_t s) {
  // Start
  cuMemAlloc(pd /*CUdeviceptr **/, s /*size_t*/);
  // End
}
