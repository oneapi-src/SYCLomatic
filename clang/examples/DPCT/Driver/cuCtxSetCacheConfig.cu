// Migration desc: The API is Removed because SYCL currently does not support.
void test(CUfunc_cache f) {
  // Start
  cuCtxSetCacheConfig(f /*CUfunc_cache*/);
  // End
}
