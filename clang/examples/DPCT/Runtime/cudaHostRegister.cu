// Migration desc: The API is Removed.
void test(void *pv, size_t s, unsigned int u) {
  // Start
  cudaHostRegister(pv /*void **/, s /*size_t*/, u /*unsigned int*/);
  // End
}
