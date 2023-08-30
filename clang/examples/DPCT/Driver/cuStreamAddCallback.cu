void test(CUstreamCallback sc, void *pv, unsigned int u) {
  // Start
  CUstream s;
  cuStreamAddCallback(s /*CUstream*/, sc /*CUstreamCallback*/, pv /*void **/,
                      u /*unsigned int*/);
  // End
}
