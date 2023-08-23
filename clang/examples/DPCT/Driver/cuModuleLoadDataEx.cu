void test(CUmodule *pm, const void *pv, unsigned int u, CUjit_option *pj,
          void **ppv) {
  // Start
  cuModuleLoadDataEx(pm /*CUmodule **/, pv /*const void **/, u /*unsigned int*/,
                     pj /*CUjit_option **/, ppv /*void ***/);
  // End
}
