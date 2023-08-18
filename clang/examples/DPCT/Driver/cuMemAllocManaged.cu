void test(CUdeviceptr *pd, size_t s, unsigned int u) {
  // Start
  cuMemAllocManaged(pd /*CUdeviceptr **/, s /*size_t*/, u /*unsigned int*/);
  // End
}
