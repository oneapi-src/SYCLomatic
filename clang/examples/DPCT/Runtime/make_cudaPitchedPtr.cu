void test(void *ptr, size_t s1, size_t s2, size_t s3) {
  // Start
  make_cudaPitchedPtr(ptr /*void **/, s1 /*size_t*/, s2 /*size_t*/,
                      s3 /*size_t*/);
  // End
}
