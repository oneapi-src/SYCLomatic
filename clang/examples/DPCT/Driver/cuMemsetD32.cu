void test(CUdeviceptr d, unsigned u, size_t s) {
  // Start
  cuMemsetD32(d /*CUdeviceptr*/, u /*unsigned*/, s /*size_t*/);
  // End
}
