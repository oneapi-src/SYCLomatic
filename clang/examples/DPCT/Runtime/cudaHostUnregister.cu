// Migration desc: The API is Removed.
void test(void *pv) {
  // Start
  cudaHostUnregister(pv /*void **/);
  // End
}
