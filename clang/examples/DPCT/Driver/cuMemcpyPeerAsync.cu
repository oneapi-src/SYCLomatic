void test(CUdeviceptr pd1, CUcontext c1, CUdeviceptr pd2, CUcontext c2,
          size_t s, CUstream cs) {
  // Start
  cuMemcpyPeerAsync(pd1 /*CUdeviceptr*/, c1 /*CUcontext*/, pd2 /*CUdeviceptr*/,
                    c2 /*CUcontext*/, s /*size_t*/, cs /*CUstream*/);
  // End
}
