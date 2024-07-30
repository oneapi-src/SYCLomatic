void test(CUdeviceptr d, unsigned short us, size_t s) {
  // Start
  cuMemsetD16(d /*CUdeviceptr*/, us /*unsigned short*/, s /*size_t*/);
  // End
}
