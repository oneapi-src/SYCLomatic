// Migration desc: The API is Removed.
void test(CUfunction f, CUfunc_cache fc) {
  // Start
  cuFuncSetCacheConfig(f /*CUfunction*/, fc /*CUfunc_cache*/);
  // End
}
