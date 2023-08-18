void test(CUdeviceptr *pd, size_t *ps, size_t s1, size_t s2, unsigned int u) {
  // Start
  cuMemAllocPitch(pd /*CUdeviceptr **/, ps /*size_t **/, s1 /*size_t*/,
                  s2 /*size_t*/, u /*unsigned int*/);
  // End
}
