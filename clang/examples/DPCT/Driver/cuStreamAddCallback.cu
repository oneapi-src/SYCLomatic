void test(CUstream s, CUstreamCallback sc, void *pv, unsigned int u) {
  // Start
  cuStreamAddCallback(s /*CUstream*/, sc /*CUstreamCallback*/, pv /*void **/,
                      u /*unsigned int*/);
  // End
}
