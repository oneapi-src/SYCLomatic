void test(void *pv, size_t s1, size_t s2, size_t s3) {
  // Start
  make_cudaPitchedPtr(pv /*void **/, s1 /*size_t*/, s2 /*size_t*/,
                      s3 /*size_t*/);
  // End
}
