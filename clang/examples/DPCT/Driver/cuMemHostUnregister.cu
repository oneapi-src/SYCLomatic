// Migration desc: The API is Removed because SYCL currently does not support.
void test(void *pv) {
  // Start
  cuMemHostUnregister(pv /*void **/);
  // End
}
