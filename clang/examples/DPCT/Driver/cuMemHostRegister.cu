// Migration desc: The API is Removed because SYCL currently does not support.
void test(void *pv, size_t s, unsigned int u) {
  // Start
  cuMemHostRegister(pv /*void **/, s /*size_t*/, u /*unsigned int*/);
  // End
}
