void test(CUdeviceptr pDev, const void *pHost, size_t s) {
  // Start
  CUstream stream;
  cuMemcpyHtoDAsync(pDev /*CUdeviceptr*/, pHost /*const void **/, s /*size_t*/,
                    stream /*CUstream*/);
  // End
}
