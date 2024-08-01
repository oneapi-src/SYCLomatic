void test(CUdeviceptr d, size_t s1, unsigned char uc, size_t s2, size_t s3,
          CUstream cs) {
  // Start
  cuMemsetD2D8Async(d /*CUdeviceptr*/, s1 /*size_t*/, uc /*unsigned char*/,
                    s2 /*size_t*/, s3 /*size_t*/, cs /*CUstream*/);
  // End
}
