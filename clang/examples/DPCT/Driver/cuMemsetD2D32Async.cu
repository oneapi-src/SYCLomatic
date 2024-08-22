void test(CUdeviceptr d, size_t s1, unsigned u, size_t s2, size_t s3,
          CUstream cs) {
  // Start
  cuMemsetD2D32Async(d /*CUdeviceptr*/, s1 /*size_t*/, u /*unsigned*/,
                     s2 /*size_t*/, s3 /*size_t*/, cs /*CUstream*/);
  // End
}
