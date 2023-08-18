void test(CUdeviceptr pd1, CUdeviceptr pd2, size_t s) {
  // Start
  cuMemcpyDtoD(pd1 /*CUdeviceptr*/, pd2 /*CUdeviceptr*/, s /*size_t*/);
  // End
}
