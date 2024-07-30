void test(size_t *ps) {
  // Start
  cuCtxGetLimit(ps /*size_t **/, CU_LIMIT_PRINTF_FIFO_SIZE);
  // End
}
