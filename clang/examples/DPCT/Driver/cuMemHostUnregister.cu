// Migration desc: The API is Removed.
void test(void *pHost) {
  // Start
  cuMemHostUnregister(pHost /*void **/);
  // End
}
