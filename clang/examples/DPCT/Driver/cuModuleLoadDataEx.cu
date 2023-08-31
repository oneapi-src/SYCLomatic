void test(CUmodule *pm, const void *pData, unsigned int u, CUjit_option *pOpt,
          void **pOptVal) {
  // Start
  cuModuleLoadDataEx(pm /*CUmodule **/, pData /*const void **/,
                     u /*unsigned int*/, pOpt /*CUjit_option **/,
                     pOptVal /*void ***/);
  // End
}
