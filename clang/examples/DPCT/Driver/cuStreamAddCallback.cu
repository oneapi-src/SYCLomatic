void test(CUstreamCallback sc, void *pData, unsigned int u) {
  // Start
  CUstream s;
  cuStreamAddCallback(s /*CUstream*/, sc /*CUstreamCallback*/, pData /*void **/,
                      u /*unsigned int*/);
  // End
}
