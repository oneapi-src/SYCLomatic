// Migration desc: The API is Removed.
void test(void *pHost, size_t s, unsigned int u) {
  // Start
  cuMemHostRegister(pHost /*void **/, s /*size_t*/, u /*unsigned int*/);
  // End
}
