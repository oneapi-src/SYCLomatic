// Migration desc: The API is Removed.
void test(cudaStream_t s, void *pv, size_t st, unsigned int u) {
  // Start
  cudaStreamAttachMemAsync(s /*cudaStream_t*/, pv /*void **/, st /*size_t*/,
                           u /*unsigned int*/);
  // End
}
