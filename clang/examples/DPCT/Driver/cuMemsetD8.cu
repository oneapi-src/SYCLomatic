void test(CUdeviceptr d, unsigned char uc, size_t s) {
  // Start
  cuMemsetD8(d /*CUdeviceptr*/, uc /*unsigned char*/, s /*size_t*/);
  // End
}
