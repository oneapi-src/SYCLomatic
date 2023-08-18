void test(CUdeviceptr d1, CUdeviceptr d2, size_t s) {
  // Start
  cuMemcpy(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/);
  // End
}
