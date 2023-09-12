void test(size_t s1, size_t s2, size_t s3) {
  // Start
  CUarray a1;
  CUarray a2;
  cuMemcpyAtoA(a1 /*CUarray*/, s1 /*size_t*/, a2 /*CUarray*/, s2 /*size_t*/,
               s3 /*size_t*/);
  // End
}
