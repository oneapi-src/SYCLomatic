void test(CUdeviceptr pd, size_t s, CUmem_advise m, CUdevice d) {
  // Start
  cuMemAdvise(pd /*CUdeviceptr*/, s /*size_t*/, m /*CUmem_advise*/,
              d /*CUdevice*/);
  // End
}
