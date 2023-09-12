// Migration desc: The API is Removed.
void test(void *pHost) {
  // Start
  cudaHostUnregister(pHost /*void **/);
  // End
}
